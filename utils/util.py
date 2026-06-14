from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import shutil
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from utils.scheduler import setup_scheduler
from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def model_has_non_finite_tensors(model):
    for _, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            return True
    return False


def get_batchnorm_state_names(model):
    state_names = set()
    base_model = model.module if hasattr(model, 'module') else model
    for module_name, module in base_model.named_modules():
        if isinstance(module, _BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            for state_name in module.state_dict().keys():
                state_names.add(prefix + state_name)
    return state_names


def clone_state_subset_cpu(state_dict, allowed_names):
    return {
        name: state_dict[name].detach().cpu().clone()
        for name in allowed_names
        if name in state_dict
    }


def load_state_dict_with_mode(args, model, state_dict, batchnorm_state_names=None):
    if getattr(args, 'fl_method', 'fedavg') != 'fedbn':
        model.load_state_dict(state_dict, strict=True)
        return

    if batchnorm_state_names is None:
        batchnorm_state_names = get_batchnorm_state_names(model)

    merged_state = model.state_dict()
    for name, tensor in state_dict.items():
        if name in batchnorm_state_names:
            continue
        merged_state[name] = tensor.detach().cpu().clone()
    model.load_state_dict(merged_state, strict=True)


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)





def inner_valid(args, model, test_loader):
    
    eval_losses = AverageMeter()

    print("++++++ Running Validation of client", args.single_client, "++++++")
    model.eval()
    all_preds, all_label = [], []
    correct_samples = 0  # 当前客户端的预测正确样本数量
    total_samples = 0   # 当前客户端的样本总数

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
        x, y = batch
        # GPU resize: 32×32 → 224×224
        if x.shape[-1] != args.img_size:
            x = F.interpolate(x, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
        total_samples += y.size(0)
        with torch.no_grad():
            logits = model(x)
            if not torch.isfinite(logits).all():
                print(f"Non-finite logits detected during validation for client {args.single_client}")
                model.train()
                eval_losses.val = float('inf')
                eval_losses.avg = float('inf')
                return 0.0, eval_losses

            if args.num_classes > 1:
                eval_loss = loss_fct(logits, y)
                if not torch.isfinite(eval_loss):
                    print(f"Non-finite validation loss detected for client {args.single_client}")
                    model.train()
                    eval_losses.val = float('inf')
                    eval_losses.avg = float('inf')
                    return 0.0, eval_losses
                eval_losses.update(eval_loss.item())
                preds = torch.argmax(logits, dim=-1)  # 分类任务
            else:
                preds = logits  # 回归任务

        # 更新当前客户端的预测正确的样本数量
        if args.num_classes > 1:  # 分类任务
            correct_samples += (preds == y).sum().item()

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    if not args.num_classes == 1:
        eval_result = simple_accuracy(all_preds, all_label)
    else:
        eval_result = mean_squared_error(all_preds, all_label)

    model.train()

    return eval_result, eval_losses


def metric_evaluation(args, eval_result):
    if args.num_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag



def valid(args, model, client_max_acc,val_loader, device, test_loader=None, TestFlag=False):
    # 验证逻辑
    eval_result, eval_losses = inner_valid(args, model, val_loader)
    print("Valid Loss: %2.5f" % eval_losses.avg, "Valid metric: %2.5f" % eval_result)

    # 记录当前客户端的最新准确率
    args.current_acc[args.single_client] = eval_result  # 确保这是单个数值（float）

    # 更新该客户端的历史最大准确率（新增逻辑）
    if args.single_client not in client_max_acc or eval_result > client_max_acc[args.single_client]:
        client_max_acc[args.single_client] = eval_result  # 直接保存浮点数值
        

    # 原有逻辑（保持CelebA和其他数据集的分支处理）
    if args.dataset == 'CelebA':
        if args.best_eval_loss[args.single_client] > eval_losses.val:
            if args.save_model_flag:
                save_model(args, model)
            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print("Updated best metric for client", args.single_client, args.best_acc[args.single_client])
            if TestFlag:
                test_result, _ = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print('Updated test acc for client', args.single_client, args.current_test_acc[args.single_client])
        else:
            print("Previous best metric remains:", args.best_acc[args.single_client])
    else:
        if metric_evaluation(args, eval_result):
            if args.save_model_flag:
                save_model(args, model)
            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print("Updated best val acc for client", args.single_client, args.best_acc[args.single_client])
            if TestFlag:
                test_result, _ = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print('Updated test acc for client', args.single_client, args.current_test_acc[args.single_client])

    return eval_result  # 返回当前客户端的准确率（float）







def optimization_fun(args, model):

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.025)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.025)

        print("===============Not implemented optimization type, we used default adamw optimizer ===============")
    return optimizer


def get_client_total_steps(args, client_name):
    if not args.dataset == 'CelebA':
        return args.clients_with_len[client_name] * args.max_communication_rounds / args.batch_size * args.E_epoch

    tmp_rounds = [math.ceil(length / 32) for length in args.clients_with_len.values()]
    denom = max(args.select_client - 1, 1)
    return sum(tmp_rounds) / denom * args.max_communication_rounds


def build_fedavg_local_optimizer_scheduler(args, model, client_name):
    optimizer = optimization_fun(args, model)
    t_total = get_client_total_steps(args, client_name)
    scheduler = (
        setup_scheduler(args, optimizer, t_total=t_total)
        if getattr(args, 'enable_lr_scheduler', True)
        else None
    )
    return optimizer, scheduler


def Partial_Client_Selection(args, model):

    # Select partial clients join in FL train
    if args.select_client == -1 or args.select_client >= len(args.dis_cvs_files): # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.select_client = len(args.dis_cvs_files)  # update the true number of selected clients
    else:
        args.proxy_clients = list(args.dis_cvs_files[:args.select_client])

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    scheduler_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    for proxy_single_client in args.proxy_clients:
        model_all[proxy_single_client] = deepcopy(model).cpu()
        optimizer_all[proxy_single_client] = None

        # get the total decay steps first
        args.t_total[proxy_single_client] = get_client_total_steps(args, proxy_single_client)
        scheduler_all[proxy_single_client] = None
        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    return model_all, optimizer_all, scheduler_all


def average_model(args, model_avg, model_all):
    model_avg.cpu()
    print('Calculate the model avg----')

    averaged_state = {}
    global_state = model_avg.state_dict()
    batchnorm_state_names = get_batchnorm_state_names(model_avg) if getattr(args, 'fl_method', 'fedavg') == 'fedbn' else set()
    valid_clients = [
        client for client in args.proxy_clients
        if not model_has_non_finite_tensors(model_all[client])
    ]

    invalid_clients = [client for client in args.proxy_clients if client not in valid_clients]
    if invalid_clients:
        print(f"Skip non-finite clients during aggregation: {invalid_clients}")

    if not valid_clients:
        print('No valid clients available for aggregation. Keeping previous global model.')
        averaged_state = {name: tensor.detach().cpu().clone() for name, tensor in global_state.items()}
        for client in args.proxy_clients:
            load_state_dict_with_mode(args, model_all[client], averaged_state, batchnorm_state_names)
        return

    total_weight = sum(float(args.clients_weightes[client]) for client in valid_clients)
    if total_weight <= 0:
        total_weight = float(len(valid_clients))

    normalized_weights = {
        client: float(args.clients_weightes[client]) / total_weight for client in valid_clients
    }
    client_states = {client: model_all[client].state_dict() for client in valid_clients}

    for name, reference in global_state.items():
        if name in batchnorm_state_names:
            averaged_state[name] = reference.detach().clone()
            continue
        if name.endswith('num_batches_tracked'):
            averaged_state[name] = torch.zeros_like(reference)
            continue

        if torch.is_floating_point(reference):
            merged = torch.zeros_like(reference, dtype=torch.float32)
            for client in valid_clients:
                client_weight = normalized_weights[client]
                merged += client_states[client][name].detach().cpu().to(torch.float32) * client_weight
            merged = torch.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
            averaged_state[name] = merged.to(reference.dtype)
        else:
            averaged_state[name] = reference.detach().clone()

    model_avg.load_state_dict(averaged_state, strict=True)

    print('Update each client model parameters----')

    for client in args.proxy_clients:
        load_state_dict_with_mode(args, model_all[client], averaged_state, batchnorm_state_names)


def clean_checkpoints_and_cache(root_dir, folder_names):
    """
    递归删除指定目录下的特定名称文件夹
    :param root_dir: 要清理的根目录
    :param folder_names: 需要删除的文件夹名称列表
    """
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname in folder_names:
                folder_path = os.path.join(dirpath, dirname)
                print(f"正在删除: {folder_path}")
                shutil.rmtree(folder_path, ignore_errors=True)
