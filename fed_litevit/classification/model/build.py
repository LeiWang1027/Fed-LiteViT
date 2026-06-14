'''
Build the FedLiteViT model family
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fed_litevit_model import FedLiteViT as FedLiteViT_MambaV23
from .old_fed_litevit import FedLiteViT as OldFedLiteViT
from .std_vit import build_std_vit
from timm.models import register_model

# v2-3 configuration
fed_litevit_cfg = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],
        'depth': [1, 1, 1],
        'num_heads': [4, 4,4],
        'window_size': [7, 7,7],
        'kernels': [3,3, 3,3],
    }


FedLiteViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

FedLiteViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

FedLiteViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

FedLiteViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

FedLiteViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
    'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }


def _load_pretrained_if_needed(model, pretrained):
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    return model


def _merge_model_cfg(model_cfg, model_overrides=None):
    merged_cfg = dict(model_cfg)
    if model_overrides:
        for key, value in model_overrides.items():
            if key in merged_cfg:
                merged_cfg[key] = value
    return merged_cfg


def _build_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, use_parallel_attention=False, use_global_mfe=True, branch_mode='both', micla_right_groups=8, model_overrides=None):
    merged_cfg = _merge_model_cfg(model_cfg, model_overrides)
    model = FedLiteViT_MambaV23(
        num_classes=num_classes,
        distillation=distillation,
        use_parallel_attention=use_parallel_attention,
        use_global_mfe=use_global_mfe,
        branch_mode=branch_mode,
        micla_right_groups=micla_right_groups,
        **merged_cfg,
    )
    model = _load_pretrained_if_needed(model, pretrained)
    if fuse:
        replace_batchnorm(model)
    return model


def _build_old_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=None):
    merged_cfg = _merge_model_cfg(model_cfg, model_overrides)
    model = OldFedLiteViT(
        num_classes=num_classes,
        distillation=distillation,
        **merged_cfg,
    )
    model = _load_pretrained_if_needed(model, pretrained)
    if fuse:
        replace_batchnorm(model)
    return model


@register_model
def fed_litevit(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=kwargs)


@register_model
def fed_litevit_Parallel(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        use_parallel_attention=True,
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_LeftOnly(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        branch_mode='left_only',
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_RightOnly(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        branch_mode='right_only',
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_RightG1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        micla_right_groups=1,
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_NoBranch(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        branch_mode='none',
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_GlobalNoMFE(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        use_global_mfe=False,
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_Parallel_LeftOnly(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        use_parallel_attention=True,
        branch_mode='left_only',
        model_overrides=kwargs,
    )


@register_model
def fed_litevit_Parallel_RightOnly(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=fed_litevit_cfg, **kwargs):
    return _build_fed_litevit_variant(
        num_classes,
        pretrained,
        distillation,
        fuse,
        model_cfg,
        use_parallel_attention=True,
        branch_mode='right_only',
        model_overrides=kwargs,
    )

@register_model
def FedLiteViT_M1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=FedLiteViT_m1, **kwargs):
    return _build_old_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=kwargs)

@register_model
def FedLiteViT_M2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=FedLiteViT_m2, **kwargs):
    return _build_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=kwargs)

@register_model
def FedLiteViT_M3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=FedLiteViT_m3, **kwargs):
    return _build_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=kwargs)

@register_model
def FedLiteViT_M4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=FedLiteViT_m4, **kwargs):
    return _build_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=kwargs)

@register_model
def FedLiteViT_M5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=FedLiteViT_m5, **kwargs):
    return _build_fed_litevit_variant(num_classes, pretrained, distillation, fuse, model_cfg, model_overrides=kwargs)


@register_model
def std_vit_6b(num_classes=1000, pretrained=False, img_size=32, pretrained_cfg=None, **kwargs):
    if pretrained:
        raise NotImplementedError('std_vit_6b does not provide pretrained weights in this project.')
    return build_std_vit(num_classes=num_classes, image_size=img_size, depth=6, **kwargs)


@register_model
def std_vit_8b(num_classes=1000, pretrained=False, img_size=32, pretrained_cfg=None, **kwargs):
    if pretrained:
        raise NotImplementedError('std_vit_8b does not provide pretrained weights in this project.')
    return build_std_vit(num_classes=num_classes, image_size=img_size, depth=8, **kwargs)

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

_checkpoint_url_format = \
    'https://github.com/xinyuliu-jeffrey/FedLiteViT_Model_Zoo/releases/download/v1.0/{}.pth'
