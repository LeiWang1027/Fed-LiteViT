import json
import os
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


RIGHT_BRANCH_PATTERNS = ["right_path"]
LEFT_BRANCH_PATTERNS = ["left_path", "mixer"]


def _get_patterns(branch):
    if branch == "right":
        return RIGHT_BRANCH_PATTERNS
    if branch == "left":
        return LEFT_BRANCH_PATTERNS
    raise ValueError(f"Unknown branch: {branch}")


def get_branch_params(model, branch="left"):
    patterns = _get_patterns(branch)
    params = []
    for name, param in model.named_parameters():
        lower_name = name.lower()
        if any(pattern in lower_name for pattern in patterns):
            params.append(param.detach().cpu().flatten())

    if not params:
        raise RuntimeError(
            f"No parameters found for branch '{branch}'. Available names:\n"
            + "\n".join(name for name, _ in model.named_parameters())
        )
    return torch.cat(params)


def has_branch_params(model, branch="left"):
    patterns = _get_patterns(branch)
    for name, _ in model.named_parameters():
        lower_name = name.lower()
        if any(pattern in lower_name for pattern in patterns):
            return True
    return False


def supports_dual_branch_analysis(model):
    return has_branch_params(model, "left") and has_branch_params(model, "right")


def get_branch_param_dict(model, branch="left"):
    patterns = _get_patterns(branch)
    result = {}
    for name, param in model.named_parameters():
        lower_name = name.lower()
        if any(pattern in lower_name for pattern in patterns):
            result[name] = param.detach().cpu().clone()
    return result


def compute_l2_distance(vec1, vec2):
    return torch.norm(vec1 - vec2, p=2).item()


def compute_normalized_l2_distance(vec1, vec2):
    dim = vec1.numel()
    if dim == 0:
        return 0.0
    return compute_l2_distance(vec1, vec2) / math.sqrt(dim)


def compute_cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()


def compute_cosine_distance(vec1, vec2):
    return 1.0 - compute_cosine_similarity(vec1, vec2)


def get_branch_param_count(model, branch="left"):
    return int(get_branch_params(model, branch).numel())


def linear_cka(x, y):
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    hsic_xy = torch.norm(x.T @ y, p="fro") ** 2
    hsic_xx = torch.norm(x.T @ x, p="fro") ** 2
    hsic_yy = torch.norm(y.T @ y, p="fro") ** 2

    denom = math.sqrt(hsic_xx.item() * hsic_yy.item())
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def get_fixed_test_subset(dataset, num_samples=1000, num_classes=100, seed=42, batch_size=64):
    rng = torch.Generator().manual_seed(seed)

    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[int(label)].append(idx)

    per_class = max(num_samples // num_classes, 1)
    selected = []
    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]
        if len(indices) > per_class:
            perm = torch.randperm(len(indices), generator=rng)[:per_class].tolist()
            chosen = [indices[i] for i in perm]
        else:
            chosen = indices
        selected.extend(chosen)

    subset = Subset(dataset, selected)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def find_target_analysis_block(model, target_block_name=None):
    target_block = None
    target_name = None
    if target_block_name is not None:
        for name, module in model.named_modules():
            if name == target_block_name:
                target_block = module
                target_name = name
                break
    else:
        for name, module in model.named_modules():
            if hasattr(module, "enable_feature_cache") and hasattr(module, "get_cached_features"):
                target_block = module
                target_name = name

    if target_block is None:
        raise RuntimeError("Cannot find target analysis block with feature cache support.")
    return target_name, target_block


def supports_feature_cache_analysis(model, target_block_name=None):
    try:
        find_target_analysis_block(model, target_block_name)
        return True
    except RuntimeError:
        return False


def extract_branch_features(model, dataloader, device, target_block_name=None, img_size=None):
    analysis_model = model.module if hasattr(model, "module") else model
    was_training = analysis_model.training
    analysis_model.eval()
    analysis_model.to(device)

    target_name, target_block = find_target_analysis_block(analysis_model, target_block_name)
    target_block.enable_feature_cache()

    all_left = []
    all_right = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            if img_size is not None and images.shape[-1] != img_size:
                images = F.interpolate(images, size=(img_size, img_size), mode="bilinear", align_corners=False)
            _ = analysis_model(images)
            left_feat, right_feat = target_block.get_cached_features()
            if left_feat is None or right_feat is None:
                raise RuntimeError(f"No cached features captured from block {target_name}")
            all_left.append(left_feat.flatten(1))
            all_right.append(right_feat.flatten(1))

    target_block.disable_feature_cache()
    analysis_model.cpu()
    if was_training:
        analysis_model.train()

    return torch.cat(all_left, dim=0), torch.cat(all_right, dim=0), target_name

