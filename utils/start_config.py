import os
import sys
import random
import hashlib
import numpy as np
import fed_litevit.classification.model.build  # noqa: F401

import torch
import torch.nn as nn
import torchvision.models as torch_models
from models import build_model
from models.cct import cct_4_2
from models.cifar_resnet import resnet32_cifar
from models.ondev_lct import ondev_lct_4_1, ondev_lct_8_1
from torch.nn import Linear
from utils.config_swin import get_config
from timm.models import create_model


FED_LITEVIT_MODEL_ALIASES = {
    'fed-litevit': 'fed_litevit',
    'fed-litevit_Parallel': 'fed_litevit_Parallel',
    'fed-litevit_LeftOnly': 'fed_litevit_LeftOnly',
    'fed-litevit_RightOnly': 'fed_litevit_RightOnly',
    'fed-litevit_RightG1': 'fed_litevit_RightG1',
    'fed-litevit_NoBranch': 'fed_litevit_NoBranch',
    'fed-litevit_GlobalNoMFE': 'fed_litevit_GlobalNoMFE',
    'fed-litevit_Parallel_LeftOnly': 'fed_litevit_Parallel_LeftOnly',
    'fed-litevit_Parallel_RightOnly': 'fed_litevit_Parallel_RightOnly',
}

WINDOWS_OUTPUT_DIR_LIMIT = 200


def _resolve_model_name_alias(model_name):
    return FED_LITEVIT_MODEL_ALIASES.get(model_name, model_name)


def _torchvision_weights(weight_enum_name, use_pretrained):
    if not use_pretrained:
        return None
    return getattr(torch_models, weight_enum_name).DEFAULT


def _parse_gpu_ids(gpu_ids):
    if isinstance(gpu_ids, (list, tuple)):
        parsed_ids = [int(gpu_id) for gpu_id in gpu_ids]
    else:
        parsed_ids = [int(gpu_id.strip()) for gpu_id in str(gpu_ids).split(',') if gpu_id.strip()]

    if not parsed_ids:
        raise ValueError("gpu_ids must contain at least one CUDA device id")
    return parsed_ids


def _require_rtx_5080(gpu_ids):
    missing_5080_ids = []
    for gpu_id in gpu_ids:
        device_name = torch.cuda.get_device_name(gpu_id)
        if 'RTX 5080' not in device_name:
            missing_5080_ids.append((gpu_id, device_name))

    if missing_5080_ids:
        details = ', '.join(
            f'cuda:{gpu_id}={device_name}' for gpu_id, device_name in missing_5080_ids
        )
        raise RuntimeError(
            'RTX 5080 is required and the selected CUDA device does not match: '
            f'{details}. Refusing to fall back to CPU or another GPU.'
        )


def _configure_gpu_runtime(args):
    available_gpu_count = torch.cuda.device_count()
    args.gpu_id_list = _parse_gpu_ids(args.gpu_ids)

    invalid_ids = [gpu_id for gpu_id in args.gpu_id_list if gpu_id < 0 or gpu_id >= available_gpu_count]
    if invalid_ids:
        raise ValueError(
            f"Invalid gpu_ids {invalid_ids}. Visible CUDA device count: {available_gpu_count}."
        )

    requested_num_gpus = int(getattr(args, 'num_gpus', 1))
    if requested_num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    if requested_num_gpus > len(args.gpu_id_list):
        raise ValueError(
            f"num_gpus={requested_num_gpus} exceeds the number of provided gpu_ids={args.gpu_id_list}"
        )

    args.num_gpus = requested_num_gpus
    args.active_gpu_ids = args.gpu_id_list[:args.num_gpus]
    args.use_distributed = bool(getattr(args, 'distributed_train', False) and args.num_gpus > 1)

    if args.use_distributed:
        local_rank = int(getattr(args, 'local_rank', 0))
        if local_rank < 0 or local_rank >= args.num_gpus:
            raise ValueError(f"Invalid local_rank={local_rank} for num_gpus={args.num_gpus}")
        args.current_gpu_id = args.active_gpu_ids[local_rank]
    else:
        args.current_gpu_id = args.active_gpu_ids[0]

    args.device = torch.device(f"cuda:{args.current_gpu_id}")
    args.use_data_parallel = args.num_gpus > 1 and not args.use_distributed
    torch.cuda.set_device(args.current_gpu_id)
    _require_rtx_5080(args.active_gpu_ids)


def _move_model_to_runtime_device(args, model):
    model = model.to(args.device)
    if getattr(args, 'use_distributed', False):
        print(f"Using distributed worker on GPU: {args.current_gpu_id}")
    elif args.use_data_parallel:
        model = nn.DataParallel(model, device_ids=args.active_gpu_ids, output_device=args.active_gpu_ids[0])
        print(f"Using DataParallel on GPUs: {args.active_gpu_ids}")
    else:
        print(f"Using single GPU: {args.current_gpu_id}")
    return model


def _compose_experiment_name(args):
    def _sanitize_name_part(value):
        return str(value).replace('/', '_').replace(os.sep, '_')

    if args.dataset in ['femnist', 'femnist-v2']:
        split_tag = f'natural_clients_{args.num_clients}_select_{args.select_client}'
    elif hasattr(args, 'alpha'):
        split_tag = f'alpha_{args.alpha}_clients_{args.num_clients}_select_{args.select_client}'
    else:
        split_tag = getattr(args, 'split_type', 'default')

    model_tag = _sanitize_name_part(getattr(args, 'model', ''))
    name_parts = [
        _sanitize_name_part(args.net_name),
        model_tag,
        getattr(args, 'fl_method', 'fedavg'),
        split_tag,
        'lr', str(args.learning_rate),
        'Pretrained', str(args.Pretrained),
        'optimizer', str(args.optimizer_type),
        'WUP',
        'Round', str(args.max_communication_rounds),
        'Eepochs', str(args.E_epoch),
        'Seed', str(args.seed),
    ]

    if getattr(args, 'fl_method', 'fedavg') == 'fedprox':
        name_parts.extend(['mu', str(getattr(args, 'fedprox_mu', 0.0))])

    run_tag = str(getattr(args, 'run_tag', '')).strip()
    if run_tag:
        name_parts.extend(['Tag', run_tag])

    return '_'.join(part for part in name_parts if str(part) != '')


def _maybe_shorten_experiment_name(args, experiment_name):
    if os.name != 'nt':
        return experiment_name

    base_output_dir = os.path.abspath(
        os.path.join(args.output_dir, args.FL_platform, args.dataset)
    )
    candidate_output_dir = os.path.join(base_output_dir, experiment_name)
    if len(candidate_output_dir) <= WINDOWS_OUTPUT_DIR_LIMIT:
        return experiment_name

    digest = hashlib.sha1(candidate_output_dir.encode('utf-8')).hexdigest()[:10]
    compact_parts = [
        str(getattr(args, 'net_name', 'run')).replace('/', '_'),
        str(getattr(args, 'model', 'model')).replace('/', '_'),
        str(getattr(args, 'fl_method', 'fedavg')),
    ]

    if args.dataset in ['femnist', 'femnist-v2']:
        compact_parts.append(f'nc{args.num_clients}')
    else:
        compact_parts.append(f'a{args.alpha}')
        compact_parts.append(f'c{args.num_clients}')

    compact_parts.extend([
        f's{args.select_client}',
        f'lr{args.learning_rate}',
        f'r{args.max_communication_rounds}',
        f'e{args.E_epoch}',
        f'seed{args.seed}',
        digest,
    ])

    shortened_name = '_'.join(part for part in compact_parts if part)
    if len(os.path.join(base_output_dir, shortened_name)) > WINDOWS_OUTPUT_DIR_LIMIT:
        shortened_name = '_'.join([
            str(getattr(args, 'net_name', 'run')).replace('/', '_'),
            str(getattr(args, 'model', 'model')).replace('/', '_'),
            digest,
        ])

    print(
        'Warning: output path is too long for Windows. '
        f'Shortening experiment directory name to: {shortened_name}'
    )
    return shortened_name

def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    args.file_name = None

    print(message)


def initization_configure(args, vis= False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project requires a GPU to run.")
    _configure_gpu_runtime(args)
    requested_model_name = getattr(args, 'model', '')
    resolved_model_name = _resolve_model_name_alias(requested_model_name)


    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True   # 输入尺寸固定，开启 benchmark 加速
    torch.backends.cudnn.deterministic = False
    # 启用 TF32 加速矩阵运算（Ampere/Ada/Blackwell GPU）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
    elif args.dataset in ['femnist', 'femnist-v2']:
        args.num_classes = 62

    if args.dataset == 'tinyimagenet':
        tinyimagenet_native_size = 64
        tinyimagenet_highres_models = {
            'fed_litevit',
            'fed_litevit_Parallel',
            'fed_litevit_LeftOnly',
            'fed_litevit_RightOnly',
            'fed_litevit_RightG1',
            'fed_litevit_NoBranch',
            'fed_litevit_GlobalNoMFE',
            'fed_litevit_Parallel_LeftOnly',
            'fed_litevit_Parallel_RightOnly',
            'FedLiteViT_M1',
        }
        if resolved_model_name not in tinyimagenet_highres_models and getattr(args, 'img_size', None) != tinyimagenet_native_size:
            print(
                f"Tiny-ImageNet uses native {tinyimagenet_native_size}x{tinyimagenet_native_size} images; "
                f"overriding img_size from {args.img_size} to {tinyimagenet_native_size} for model {args.model}."
            )
            args.img_size = tinyimagenet_native_size

        if hasattr(args, 'input_size') and getattr(args, 'input_size', None) != args.img_size:
            args.input_size = args.img_size

    # 为build_transform设置默认参数（兼容未定义这些参数的训练脚本）
    _transform_defaults = {
        'input_size': getattr(args, 'img_size', 224),
        'color_jitter': 0.0,
        'aa': '',
        'reprob': 0.0,
        'remode': 'pixel',
        'recount': 0,
        'finetune': '',
        'data_set': 'MINI' if args.dataset == 'tinyimagenet' else 'CIFAR',
    }
    for attr, default in _transform_defaults.items():
        if not hasattr(args, attr):
            setattr(args, attr, default)

    # Set model type related parameters
    if "ResNet" in args.FL_platform:
        args.Use_ResNet = True
        if getattr(args, 'model', '') == 'ResNet-32':
            model = resnet32_cifar(num_classes=args.num_classes)
            print('We use ResNet 32 for CIFAR')
        elif getattr(args, 'model', '') == 'MobileNetV2':
            model = torch_models.mobilenet_v2(weights=_torchvision_weights("MobileNet_V2_Weights", args.Pretrained))
            print('We use MobileNetV2')
        elif '101' in args.net_name:
            model = torch_models.resnet152(weights=_torchvision_weights("ResNet152_Weights", args.Pretrained))
            # model.fc = nn.Linear(2048, args.num_classes)
            print('We use ResNet 152')

        elif '32_8' in args.net_name:

            model = torch_models.resnext101_32x8d(weights=_torchvision_weights("ResNeXt101_32X8D_Weights", args.Pretrained))
            print('We use ResNet 101-32*8d')

        else:
            model = torch_models.resnet50(weights=_torchvision_weights("ResNet50_Weights", args.Pretrained))
            print('We use default ResNet 50')
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            last_layer = model.classifier[-1]
            if isinstance(last_layer, nn.Linear):
                model.classifier[-1] = nn.Linear(last_layer.in_features, args.num_classes)
        model = _move_model_to_runtime_device(args, model)

    elif "ViT" in args.FL_platform:
        model_name = {
            'std-vit-6b': 'std_vit_6b',
            'std-vit-8b': 'std_vit_8b',
        }.get(resolved_model_name, resolved_model_name)
        fed_litevit_models = {
            'fed_litevit',
            'fed_litevit_Parallel',
            'fed_litevit_LeftOnly',
            'fed_litevit_RightOnly',
            'fed_litevit_RightG1',
            'fed_litevit_NoBranch',
            'fed_litevit_GlobalNoMFE',
            'fed_litevit_Parallel_LeftOnly',
            'fed_litevit_Parallel_RightOnly',
        }
        timm_vit_models = {
            'std_vit_6b',
            'std_vit_8b',
            'fed_litevit',
            'fed_litevit_Parallel',
            'fed_litevit_LeftOnly',
            'fed_litevit_RightOnly',
            'fed_litevit_RightG1',
            'fed_litevit_NoBranch',
            'fed_litevit_GlobalNoMFE',
            'fed_litevit_Parallel_LeftOnly',
            'fed_litevit_Parallel_RightOnly',
            'FedLiteViT_M1',
            'FedLiteViT_M2',
            'FedLiteViT_M3',
            'FedLiteViT_M4',
            'FedLiteViT_M5',
        }
        if model_name in timm_vit_models:
            if model_name in fed_litevit_models and args.img_size not in (32, 224):
                raise ValueError(
                    f"{args.model} supports img_size 32 or 224, got {args.img_size}"
                )
            print('We use model', args.model)
            model = create_model(
                model_name,
                num_classes=args.num_classes,
                distillation=(args.distillation_type != 'none'),
                pretrained=False,
                fuse=False,
                img_size=args.img_size,
            )
        elif model_name == 'OnDev-LCT-8/1':
            print('We use model', args.model)
            model = ondev_lct_8_1(num_classes=args.num_classes, img_size=args.img_size)
        elif model_name == 'OnDev-LCT-4/1':
            print('We use model', args.model)
            model = ondev_lct_4_1(num_classes=args.num_classes, img_size=args.img_size)
        elif model_name == 'CCT-4/2':
            print('We use model', args.model)
            model = cct_4_2(num_classes=args.num_classes, img_size=args.img_size)
        elif 'tiny' in args.net_name:
            print('We use ViT tiny')
            from timm.models.vision_transformer import vit_tiny_patch16_224

            model = vit_tiny_patch16_224(pretrained=args.Pretrained)
        elif 'small' in args.net_name:
            print('We use ViT small')
            from timm.models.vision_transformer import vit_small_patch16_224
            model = vit_small_patch16_224(pretrained=args.Pretrained)
        else:
            from timm.models.vision_transformer import vit_base_patch16_224
            print('We use default ViT settting base')
            model = vit_base_patch16_224(pretrained=args.Pretrained)
        model = _move_model_to_runtime_device(args, model)


    elif "Swin" in args.FL_platform:
        print('We use Swin')
        if not args.cfg:
            sys.exit('Network configure file cfg for Swin is missing, code is exit')
        swin_args = get_config(args)
        model = build_model(args, swin_args)
        if args.Pretrained:
            checkpoint = torch.load(args.pretrained_dir, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)

        model.head = Linear(model.head.weight.shape[1], args.num_classes)
        model = _move_model_to_runtime_device(args, model)

    # set output parameters
    print(args.optimizer_type)
    args.original_experiment_name = _compose_experiment_name(args)
    args.name = _maybe_shorten_experiment_name(args, args.original_experiment_name)


    # args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name)
    args.output_dir = os.path.join(args.output_dir, args.FL_platform, args.dataset, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    return model
