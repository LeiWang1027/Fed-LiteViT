# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP


def _resolve_compatible_window_size(img_size, patch_size, depths, requested_window_size):
    stage_resolution = img_size // patch_size
    stage_resolutions = []
    for _ in depths:
        stage_resolutions.append(stage_resolution)
        stage_resolution //= 2

    max_window_size = min(requested_window_size, min(stage_resolutions))
    for candidate in range(max_window_size, 0, -1):
        if all(resolution % candidate == 0 for resolution in stage_resolutions):
            return candidate

    return 1


def build_model(args, swin_args):
    model_type = swin_args.MODEL.TYPE
    if model_type == 'swin':
        window_size = _resolve_compatible_window_size(
            img_size=args.img_size,
            patch_size=swin_args.MODEL.SWIN.PATCH_SIZE,
            depths=swin_args.MODEL.SWIN.DEPTHS,
            requested_window_size=swin_args.MODEL.SWIN.WINDOW_SIZE,
        )
        if window_size != swin_args.MODEL.SWIN.WINDOW_SIZE:
            print(
                f"Adjust Swin window_size from {swin_args.MODEL.SWIN.WINDOW_SIZE} to {window_size} "
                f"for img_size={args.img_size}."
            )
        model = SwinTransformer(img_size=args.img_size,
                                patch_size=swin_args.MODEL.SWIN.PATCH_SIZE,
                                in_chans=swin_args.MODEL.SWIN.IN_CHANS,
                                num_classes=swin_args.MODEL.NUM_CLASSES,
                                embed_dim=swin_args.MODEL.SWIN.EMBED_DIM,
                                depths=swin_args.MODEL.SWIN.DEPTHS,
                                num_heads=swin_args.MODEL.SWIN.NUM_HEADS,
                                window_size=window_size,
                                mlp_ratio=swin_args.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=swin_args.MODEL.SWIN.QKV_BIAS,
                                qk_scale=swin_args.MODEL.SWIN.QK_SCALE,
                                drop_rate=swin_args.MODEL.DROP_RATE,
                                drop_path_rate=swin_args.MODEL.DROP_PATH_RATE,
                                ape=swin_args.MODEL.SWIN.APE,
                                patch_norm=swin_args.MODEL.SWIN.PATCH_NORM)
    elif model_type == 'swin_mlp':
        window_size = _resolve_compatible_window_size(
            img_size=args.img_size,
            patch_size=swin_args.MODEL.SWIN_MLP.PATCH_SIZE,
            depths=swin_args.MODEL.SWIN_MLP.DEPTHS,
            requested_window_size=swin_args.MODEL.SWIN_MLP.WINDOW_SIZE,
        )
        if window_size != swin_args.MODEL.SWIN_MLP.WINDOW_SIZE:
            print(
                f"Adjust Swin-MLP window_size from {swin_args.MODEL.SWIN_MLP.WINDOW_SIZE} to {window_size} "
                f"for img_size={args.img_size}."
            )
        model = SwinMLP(img_size=args.img_size,
                        patch_size=swin_args.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=swin_args.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=swin_args.MODEL.NUM_CLASSES,
                        embed_dim=swin_args.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=swin_args.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=swin_args.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=window_size,
                        mlp_ratio=swin_args.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=swin_args.MODEL.DROP_RATE,
                        drop_path_rate=swin_args.MODEL.DROP_PATH_RATE,
                        ape=swin_args.MODEL.SWIN_MLP.APE,
                        patch_norm=swin_args.MODEL.SWIN_MLP.PATCH_NORM)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
