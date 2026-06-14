from __future__ import absolute_import, division, print_function

import os
import sys
import json
import argparse
import gc
import re
import selectors
import subprocess
import tempfile
import socket
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from analysis_utils import (
    compute_cosine_distance,
    compute_l2_distance,
    compute_normalized_l2_distance,
    extract_branch_features,
    get_branch_param_count,
    get_branch_params,
    get_fixed_test_subset,
    linear_cka,
    supports_dual_branch_analysis,
    supports_feature_cache_analysis,
)
from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.util import (
    Partial_Client_Selection,
    build_fedavg_local_optimizer_scheduler,
    valid,
    average_model,
    clean_checkpoints_and_cache,
    clone_state_subset_cpu,
    get_batchnorm_state_names,
    load_state_dict_with_mode,
)
from utils.start_config import initization_configure
import warnings
warnings.filterwarnings("ignore")

DEFAULT_PARALLEL_SEEDS = (42, 43, 44)
ROUND_SUMMARY_PATTERN = re.compile(r"\[RoundSummary\].*?best_acc=([0-9.]+)")


def launch_default_seed_runs(argv, output_dir):
    processes = []
    selector = selectors.DefaultSelector()
    seed_best_acc = {seed: None for seed in DEFAULT_PARALLEL_SEEDS}

    for seed in DEFAULT_PARALLEL_SEEDS:
        command = [sys.executable, "-u", os.path.abspath(__file__), *argv, "--seed", str(seed)]
        print(f"Launching seed {seed}: {' '.join(command)}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        processes.append((seed, process))
        selector.register(process.stdout, selectors.EVENT_READ, seed)

    open_streams = len(processes)
    while open_streams:
        for key, _ in selector.select():
            line = key.fileobj.readline()
            if line:
                seed = key.data
                print(f"[seed {seed}] {line}", end="", flush=True)
                summary_match = ROUND_SUMMARY_PATTERN.search(line)
                if summary_match:
                    seed_best_acc[seed] = float(summary_match.group(1))
            else:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                open_streams -= 1

    failed = False
    for seed, process in processes:
        returncode = process.wait()
        status = "completed" if returncode == 0 else "failed"
        print(f"Seed {seed} {status}: returncode={returncode}")
        failed = failed or returncode != 0

    print("\n========== Seed Best Accuracy Summary ==========")
    for seed in DEFAULT_PARALLEL_SEEDS:
        best_acc = seed_best_acc.get(seed)
        if best_acc is None:
            print(f"seed={seed} best_acc=N/A")
        else:
            print(f"seed={seed} best_acc={best_acc:.4f}")
    print("================================================")

    if failed:
        raise SystemExit(1)

# 添加MACs计算相关导入
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not available. Please install with: pip install thop")
    THOP_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


def _build_dataloader_kwargs(num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = prefetch_factor
    return kwargs


def is_main_process(args):
    return not getattr(args, 'distributed_train', False) or getattr(args, 'rank', 0) == 0


def distributed_barrier(args):
    if getattr(args, 'distributed_train', False):
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def shutdown_dataloader_workers(data_loader):
    if data_loader is None:
        return

    iterator = getattr(data_loader, '_iterator', None)
    if iterator is None:
        return

    shutdown_workers = getattr(iterator, '_shutdown_workers', None)
    if shutdown_workers is not None:
        shutdown_workers()
    data_loader._iterator = None


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def compute_fedprox_penalty(current_model, reference_state, device):
    penalty = torch.zeros((), device=device)
    for name, param in unwrap_model(current_model).named_parameters():
        if not param.requires_grad:
            continue
        reference_param = reference_state[name].to(device=device, dtype=param.dtype)
        penalty = penalty + torch.sum((param - reference_param) ** 2)
    return penalty


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def setup_distributed(args, local_rank):
    args.local_rank = local_rank
    args.rank = local_rank
    args.world_size = args.num_gpus
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )


def configure_runtime_batch_sizes(args):
    # Client-level parallelism keeps one full local batch per GPU worker.
    args.per_gpu_batch_size = args.batch_size


def get_runtime_loader_kwargs(args):
    effective_num_workers = args.num_workers
    if getattr(args, 'distributed_train', False):
        effective_num_workers = min(effective_num_workers, 1)

    # Client-specific loaders are rebuilt repeatedly during federated rounds,
    # so keeping worker processes alive across loader instances leaks file
    # descriptors and can hit the OS open-file limit.
    persistent_workers = False
    return _build_dataloader_kwargs(
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )


def build_train_dataset_for_client(args, client_name):
    original_single_client = getattr(args, 'single_client', None)
    try:
        args.single_client = client_name
        trainset = DatasetFLViT(args, phase='train')
    finally:
        args.single_client = original_single_client

    return trainset


def build_train_loader_for_client(args, common_loader_kwargs, trainset):

    sampler = RandomSampler(trainset)
    train_loader = DataLoader(
        trainset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        drop_last=True,
        **common_loader_kwargs,
    )
    return train_loader


def select_round_clients(args, tot_clients):
    if args.select_client == len(args.dis_cvs_files):
        selected_clients = args.proxy_clients
    else:
        selected_clients = np.random.choice(tot_clients, args.select_client, replace=False).tolist()

    if getattr(args, 'distributed_train', False):
        broadcast_payload = [selected_clients]
        dist.broadcast_object_list(broadcast_payload, src=0)
        selected_clients = broadcast_payload[0]

    return selected_clients


def get_round_client_pairs(args, selected_clients):
    return list(zip(selected_clients, args.proxy_clients))


def get_local_client_pairs(args, round_client_pairs):
    if not getattr(args, 'distributed_train', False):
        return round_client_pairs
    return [
        pair for index, pair in enumerate(round_client_pairs)
        if index % args.world_size == args.rank
    ]


def build_local_training_payload(args, model_all, local_client_pairs):
    payload = {
        'model_states': {},
        'global_steps': {},
        'learning_rate_updates': {},
        'fedbn_states': {},
    }
    for cur_single_client, proxy_single_client in local_client_pairs:
        payload['model_states'][proxy_single_client] = clone_state_dict_cpu(
            model_all[proxy_single_client].state_dict()
        )
        payload['global_steps'][proxy_single_client] = int(
            args.global_step_per_client[proxy_single_client]
        )
        start_index = args.learning_rate_sync_offsets[proxy_single_client]
        payload['learning_rate_updates'][proxy_single_client] = list(
            args.learning_rate_record[proxy_single_client][start_index:]
        )
        args.learning_rate_sync_offsets[proxy_single_client] = len(
            args.learning_rate_record[proxy_single_client]
        )
        if getattr(args, 'fl_method', 'fedavg') == 'fedbn' and getattr(args, 'batchnorm_state_names', None):
            payload['fedbn_states'][cur_single_client] = clone_state_subset_cpu(
                model_all[proxy_single_client].state_dict(),
                args.batchnorm_state_names,
            )
    return payload


def merge_training_payload(args, model_all, gathered_payloads):
    for payload in gathered_payloads:
        if not payload:
            continue
        for proxy_single_client, state_dict in payload['model_states'].items():
            load_state_dict_with_mode(
                args,
                model_all[proxy_single_client],
                state_dict,
                getattr(args, 'batchnorm_state_names', None),
            )
        for proxy_single_client, global_step in payload['global_steps'].items():
            args.global_step_per_client[proxy_single_client] = int(global_step)
        for proxy_single_client, lr_updates in payload['learning_rate_updates'].items():
            if lr_updates:
                args.learning_rate_record[proxy_single_client].extend(lr_updates)
            args.learning_rate_sync_offsets[proxy_single_client] = len(args.learning_rate_record[proxy_single_client])
        for client_name, batchnorm_state in payload.get('fedbn_states', {}).items():
            args.fedbn_local_states[client_name] = batchnorm_state


def synchronize_local_training_results(args, model_all, local_client_pairs):
    local_payload = build_local_training_payload(args, model_all, local_client_pairs)
    if not getattr(args, 'distributed_train', False):
        merge_training_payload(args, model_all, [local_payload])
        return

    gathered_payloads = [None for _ in range(args.world_size)] if is_main_process(args) else None
    dist.gather_object(local_payload, gathered_payloads, dst=0)
    if is_main_process(args):
        merge_training_payload(args, model_all, gathered_payloads)


def broadcast_global_model_state(args, model_avg, model_all):
    averaged_state = clone_state_dict_cpu(model_avg.state_dict()) if is_main_process(args) else None
    if getattr(args, 'distributed_train', False):
        state_container = [averaged_state]
        dist.broadcast_object_list(state_container, src=0)
        averaged_state = state_container[0]

    model_avg.load_state_dict(averaged_state, strict=True)
    for proxy_single_client in args.proxy_clients:
        load_state_dict_with_mode(
            args,
            model_all[proxy_single_client],
            averaged_state,
            getattr(args, 'batchnorm_state_names', None),
        )


def reduce_round_gradient_stats(args, round_gradient_stats):
    reduced_stats = dict(round_gradient_stats)
    if not getattr(args, 'distributed_train', False):
        return reduced_stats

    device = args.device
    max_tensor = torch.tensor([round_gradient_stats['max_grad_norm']], dtype=torch.float32, device=device)
    sum_tensor = torch.tensor([
        round_gradient_stats['total_grad_norm'],
        round_gradient_stats['step_count'],
        round_gradient_stats['clip_count'],
    ], dtype=torch.float64, device=device)
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)

    reduced_stats['max_grad_norm'] = float(max_tensor.item())
    reduced_stats['total_grad_norm'] = float(sum_tensor[0].item())
    reduced_stats['step_count'] = int(sum_tensor[1].item())
    reduced_stats['clip_count'] = int(sum_tensor[2].item())
    return reduced_stats


def collect_round_nan_events(args, local_round_nan_events):
    if not getattr(args, 'distributed_train', False):
        return list(local_round_nan_events), len(local_round_nan_events), local_round_nan_events[0] if local_round_nan_events else None

    gathered_events = [None for _ in range(args.world_size)] if is_main_process(args) else None
    gathered_first_events = [None for _ in range(args.world_size)] if is_main_process(args) else None
    local_first_event = local_round_nan_events[0] if local_round_nan_events else None
    dist.gather_object(local_round_nan_events, gathered_events, dst=0)
    dist.gather_object(local_first_event, gathered_first_events, dst=0)

    if not is_main_process(args):
        return [], 0, None

    merged_events = []
    for events in gathered_events:
        if events:
            merged_events.extend(events)
    first_event = next((event for event in gathered_first_events if event is not None), None)
    return merged_events, len(merged_events), first_event


def maybe_release_cuda_cache(args):
    if getattr(args, 'empty_cache_per_client', False):
        torch.cuda.empty_cache()


def compute_grad_norm(model):
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5


def has_non_finite_gradients(model):
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return True
    return False


def clone_state_dict_cpu(state_dict):
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def restore_batchnorm_state(model, batchnorm_state):
    if not batchnorm_state:
        return

    model_to_update = unwrap_model(model)
    current_state = model_to_update.state_dict()
    for name, tensor in batchnorm_state.items():
        current_state[name] = tensor.detach().cpu().clone()
    model_to_update.load_state_dict(current_state, strict=True)


def state_dict_has_non_finite(state_dict):
    for tensor in state_dict.values():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            return True
    return False


def validate_fedbn_shared_loader(args, model_avg, client_max_acc, val_loader):
    available_clients = [
        client for client in args.dis_cvs_files
        if args.fedbn_local_states.get(client)
    ]

    if not available_clients:
        print('No client-specific BN states cached yet. Falling back to global-model validation.')
        args.single_client = 'global'
        args.best_acc.setdefault('global', float('-inf') if args.num_classes != 1 else float('inf'))
        args.best_eval_loss.setdefault('global', float('inf'))
        args.current_acc.setdefault('global', 0.0)
        args.current_test_acc.setdefault('global', 0.0)
        model_avg.to(args.device)
        valid(args, model_avg, client_max_acc, val_loader, None, TestFlag=False)
        model_avg.cpu()
        return

    weighted_metric_sum = 0.0
    total_weight = 0.0

    for client in available_clients:
        eval_model = deepcopy(model_avg).cpu()
        restore_batchnorm_state(eval_model, args.fedbn_local_states[client])
        eval_model.to(args.device)

        args.single_client = client
        metric = float(valid(args, eval_model, client_max_acc, val_loader, None, TestFlag=False))
        client_weight = float(args.clients_with_len.get(client, 1.0))
        weighted_metric_sum += metric * client_weight
        total_weight += client_weight

        eval_model.cpu()

    global_metric = weighted_metric_sum / total_weight if total_weight > 0 else 0.0
    args.single_client = 'global'
    args.best_acc.setdefault('global', float('-inf') if args.num_classes != 1 else float('inf'))
    args.best_eval_loss.setdefault('global', float('inf'))
    args.current_acc['global'] = global_metric
    args.current_test_acc.setdefault('global', 0.0)

    if args.num_classes == 1:
        args.best_acc['global'] = min(args.best_acc['global'], global_metric)
    else:
        args.best_acc['global'] = max(args.best_acc['global'], global_metric)

    print(
        f'FedBN shared-test weighted mean metric: {global_metric:.5f} '
        f'across {len(available_clients)} client models'
    )


def append_nan_event(nan_debug_events, stage, epoch, client_name, inner_epoch=None, step=None, loss=None, grad_norm=None, lr=None, reason=None):
    event = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'stage': stage,
        'round': int(epoch),
        'client': str(client_name),
        'inner_epoch': None if inner_epoch is None else int(inner_epoch),
        'step': None if step is None else int(step),
        'loss': None if loss is None else float(loss),
        'grad_norm': None if grad_norm is None else float(grad_norm),
        'lr': None if lr is None else float(lr),
        'reason': reason,
    }
    nan_debug_events.append(event)
    return event

def calculate_model_macs(model, input_shape=(3, 224, 224), batch_size=1, device='cuda'):
    """
    计算模型的MACs (Multiply-Accumulate Operations)
    
    Args:
        model: PyTorch模型
        input_shape: 输入张量形状 (C, H, W)
        batch_size: 批次大小
        device: 计算设备
    
    Returns:
        dict: 包含MACs、FLOPs、参数量等信息的字典
    """
    model_stats = {
        'total_params': 0,
        'trainable_params': 0,
        'macs': 0,
        'flops': 0,
        'model_size_mb': 0,
        'input_shape': input_shape,
        'batch_size': batch_size
    }
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_stats['total_params'] = total_params
    model_stats['trainable_params'] = trainable_params
    
    # 计算模型大小（MB）
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_stats['model_size_mb'] = (param_size + buffer_size) / 1024 / 1024
    
    # 创建输入张量
    print("********************************",batch_size)
    batch_size= 1
    input_tensor = torch.randn(batch_size, *input_shape).to(device)
    model = model.to(device)
    model.eval()
    
    # 方法1: 使用thop库计算MACs和FLOPs
    if THOP_AVAILABLE:
        try:
            with torch.no_grad():
                macs, params = profile(model, inputs=(input_tensor,), verbose=False)
                model_stats['macs'] = macs
                model_stats['flops'] = macs * 2  # MACs ≈ FLOPs/2
                
                # 格式化输出
                macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
                model_stats['macs_formatted'] = macs_formatted
                model_stats['params_formatted'] = params_formatted
                
                print(f"THOP Results - MACs: {macs_formatted}, Params: {params_formatted}/arg.")
        except Exception as e:
            print(f"THOP calculation failed: {e}")
    
    # 方法2: 使用fvcore库计算FLOPs
    if FVCORE_AVAILABLE:
        try:
            with torch.no_grad():
                flops_analysis = FlopCountAnalysis(model, input_tensor)
                total_flops = flops_analysis.total()
                model_stats['flops_fvcore'] = total_flops
                model_stats['macs_fvcore'] = total_flops / 2
                
                print(f"FVCore Results - FLOPs: {total_flops:,}, MACs: {total_flops/2:,}")
        except Exception as e:
            print(f"FVCore calculation failed: {e}")
    
    # 方法3: 手动计算（备用方案）
    if not THOP_AVAILABLE and not FVCORE_AVAILABLE:
        print("Warning: No FLOP counting library available. Using parameter count estimation.")
        # 简单估算：假设每个参数进行一次乘加运算
        estimated_macs = total_params * batch_size
        model_stats['macs'] = estimated_macs
        model_stats['flops'] = estimated_macs * 2
    
    return model_stats

def print_model_complexity(model_stats):
    """打印模型复杂度信息"""
    print("\n" + "="*60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*60)
    
    print(f"Input Shape: {model_stats['input_shape']}")
    print(f"Batch Size: {model_stats['batch_size']}")
    print(f"Total Parameters: {model_stats['total_params']:,}")
    print(f"Trainable Parameters: {model_stats['trainable_params']:,}")
    print(f"Model Size: {model_stats['model_size_mb']:.2f} MB")
    
    if 'macs_formatted' in model_stats:
        print(f"MACs (THOP): {model_stats['macs_formatted']}")
        print(f"Params (THOP): {model_stats['params_formatted']}")
    
    if model_stats['macs'] > 0:
        macs_g = model_stats['macs'] / 1e9
        flops_g = model_stats['flops'] / 1e9
        print(f"MACs: {macs_g:.3f} G")
        print(f"FLOPs: {flops_g:.3f} G")
    
    if 'macs_fvcore' in model_stats:
        macs_fv_g = model_stats['macs_fvcore'] / 1e9
        flops_fv_g = model_stats['flops_fvcore'] / 1e9
        print(f"MACs (FVCore): {macs_fv_g:.3f} G")
        print(f"FLOPs (FVCore): {flops_fv_g:.3f} G")
    
    print("="*60)


def prepare_run_environment(args):
    if not getattr(args, 'clean_output_cache', True):
        return

    run_output_dir = os.path.abspath(getattr(args, 'output_dir', ''))
    if run_output_dir and os.path.exists(run_output_dir):
        print(f"正在清理当前实验输出目录: {run_output_dir}")
        clean_checkpoints_and_cache(os.path.dirname(run_output_dir), [os.path.basename(run_output_dir)])

    notebook_cache_dir = os.path.abspath(".ipynb_checkpoints")
    if os.path.isdir(notebook_cache_dir):
        print(f"正在删除: {notebook_cache_dir}")
        clean_checkpoints_and_cache(os.path.dirname(notebook_cache_dir), [os.path.basename(notebook_cache_dir)])

def train(args, model):
    """ Train the model """

    if is_main_process(args):
        prepare_run_environment(args)
    distributed_barrier(args)

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = None

    model_stats = None
    if args.enable_model_complexity and is_main_process(args):
        print("\n计算模型复杂度...")
        input_shape = (3, args.img_size, args.img_size)
        model_stats = calculate_model_macs(
            model=deepcopy(model),
            input_shape=input_shape,
            batch_size=args.batch_size,
            device=args.device
        )
        print_model_complexity(model_stats)
    
    # 初始化结果记录
    round_results = [] if is_main_process(args) else None
    convergence_history = [] if is_main_process(args) else None
    client_max_acc = {} if is_main_process(args) else None
    
    # 添加梯度监控变量
    gradient_stats = {
        'max_grad_norms': [],
        'avg_grad_norms': [],
        'grad_clip_count': 0
    }
    nan_debug_events = []

    # Prepare dataset
    args.enable_cifar_tensor_cache = not getattr(args, 'disable_cifar_tensor_cache', False)
    create_dataset_and_evalmetrix(args)

    common_loader_kwargs = get_runtime_loader_kwargs(args)
    train_dataset_cache = {}
    if getattr(args, 'distributed_train', False) and is_main_process(args):
        print(
            f"Distributed mode detected: reducing DataLoader workers from {args.num_workers} "
            f"to {common_loader_kwargs['num_workers']} per process"
        )
    val_loader = None
    fixed_test_loader = None
    if is_main_process(args):
        valset = DatasetFLViT(args, phase='val')
        val_loader = DataLoader(
            valset,
            sampler=SequentialSampler(valset),
            batch_size=args.batch_size,
            **common_loader_kwargs,
        )
    analysis_results_dir = os.path.join(args.output_dir, "results")
    if is_main_process(args):
        os.makedirs(analysis_results_dir, exist_ok=True)
    divergence_log = [] if is_main_process(args) else None
    cka_log = [] if is_main_process(args) else None
    target_analysis_block_name = None

    # Configuration for FedAVG
    model_all, optimizer_all, scheduler_all = Partial_Client_Selection(args, model)
    args.learning_rate_sync_offsets = {proxy_single_client: 0 for proxy_single_client in args.proxy_clients}
    model_avg = deepcopy(model).cpu()
    args.batchnorm_state_names = get_batchnorm_state_names(model_avg) if args.fl_method == 'fedbn' else set()
    args.fedbn_local_states = {}
    if args.fl_method == 'fedbn' and not args.batchnorm_state_names:
        print('FedBN selected, but the current model has no BatchNorm layers. Aggregation will match FedAvg behavior.')
    extra_analysis_enabled = bool(getattr(args, 'enable_extra_analysis', False))
    exp1_enabled = (
        is_main_process(args)
        and extra_analysis_enabled
        and supports_dual_branch_analysis(model_avg)
    )
    exp2_enabled = (
        is_main_process(args)
        and exp1_enabled
        and supports_feature_cache_analysis(model_avg)
    )
    if is_main_process(args) and not extra_analysis_enabled:
        print("Extra experiment analysis disabled; running training/aggregation/validation only.")
    elif is_main_process(args) and not exp1_enabled:
        print("Skip experiment 1 analysis: model does not expose left/right branch parameters.")
    if is_main_process(args) and extra_analysis_enabled and not exp2_enabled:
        print("Skip experiment 2 analysis: model does not expose dual-branch feature-cache analysis support.")
    if exp2_enabled:
        fixed_test_loader = get_fixed_test_subset(
            valset,
            num_samples=min(args.cka_num_samples, len(valset)),
            num_classes=args.num_classes,
            seed=args.seed,
            batch_size=64,
        )
    total_params = sum(p.numel() for p in model_avg.parameters())
    print(f"Total Model Parameters: {total_params}")

    # Train!
    print("=============== Running training ===============")
    print(f"Gradient clipping enabled: {args.grad_clip}")
    print(f"Max gradient norm: {args.max_grad_norm}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LR scheduler enabled: {args.enable_lr_scheduler}")
    print(f"Data augmentation enabled: {args.enable_data_augmentation}")

    loss_fct = torch.nn.CrossEntropyLoss()
    tot_clients = args.dis_cvs_files
    epoch = 0
    training_start_time = datetime.now()
    max_acc = 0.0
    max_client_name = "N/A"

    while epoch < args.max_communication_rounds:
        print(f"\n========== Round {epoch}/{args.max_communication_rounds} ==========")
        local_round_nan_events = []
        round_non_finite_events = 0
        first_non_finite_event = None
        
        # 记录本轮开始时间
        round_start_time = datetime.now()
        
        # Client selection
        cur_selected_clients = select_round_clients(args, tot_clients)
        round_client_pairs = get_round_client_pairs(args, cur_selected_clients)
        local_client_pairs = get_local_client_pairs(args, round_client_pairs)
        if local_client_pairs:
            assigned_clients = ', '.join(
                f"{cur_single_client}->{proxy_single_client}" for cur_single_client, proxy_single_client in local_client_pairs
            )
        else:
            assigned_clients = 'none'
        print(f"Rank {getattr(args, 'rank', 0)} assigned clients: {assigned_clients}")

        # 计算客户端权重
        cur_tot_client_Lens = sum(args.clients_with_len[client] for client in cur_selected_clients)
        for cur_single_client, proxy_single_client in round_client_pairs:
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens

        # 每轮训练的梯度统计
        round_gradient_stats = {
            'max_grad_norm': 0,
            'total_grad_norm': 0,
            'step_count': 0,
            'clip_count': 0
        }

        # Client training loop
        for cur_single_client, proxy_single_client in local_client_pairs:
            args.single_client = cur_single_client

            if cur_single_client not in train_dataset_cache:
                train_dataset_cache[cur_single_client] = build_train_dataset_for_client(
                    args,
                    cur_single_client,
                )
            train_loader = build_train_loader_for_client(
                args,
                common_loader_kwargs,
                train_dataset_cache[cur_single_client],
            )
            client_last_loss = None
            client_last_grad_norm = None
            client_last_lr = None
            client_step_count = 0

            # 模型训练
            if args.fl_method == 'fedbn' and args.batchnorm_state_names:
                restore_batchnorm_state(
                    model_all[proxy_single_client],
                    args.fedbn_local_states.get(cur_single_client),
                )
            model = model_all[proxy_single_client].to(args.device).train()
            optimizer, scheduler = build_fedavg_local_optimizer_scheduler(
                args,
                model,
                cur_single_client,
            )
            proximal_reference_state = None
            if args.fl_method == 'fedprox':
                proximal_reference_state = {
                    name: tensor.detach().clone()
                    for name, tensor in unwrap_model(model).state_dict().items()
                    if torch.is_floating_point(tensor)
                }

            print(
                f'Training client {cur_single_client} on rank {getattr(args, "rank", 0)} '
                f'| Proxy {proxy_single_client} | Round {epoch}/{args.max_communication_rounds}'
            )
            
            # 本地训练周期
            for inner_epoch in range(args.E_epoch):
                for step, batch in enumerate(train_loader):
                    args.global_step_per_client[proxy_single_client] += 1
                    batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
                    x, y = batch

                    # GPU resize: 32×32 → 224×224
                    if x.shape[-1] != args.img_size:
                        x = F.interpolate(x, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)

                    # 前向传播和反向传播
                    optimizer.zero_grad(set_to_none=True)
                    predict = model(x)
                    if not torch.isfinite(predict).all():
                        event = append_nan_event(
                            local_round_nan_events,
                            stage='forward',
                            epoch=epoch,
                            client_name=cur_single_client,
                            inner_epoch=inner_epoch,
                            step=step,
                            lr=optimizer.param_groups[0]['lr'],
                            reason='non-finite logits',
                        )
                        round_non_finite_events += 1
                        if first_non_finite_event is None:
                            first_non_finite_event = event
                            print(f"[NonFinite][Round {epoch}] client={cur_single_client} inner_epoch={inner_epoch} step={step} reason={event['reason']} lr={event['lr']:.6f}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss = loss_fct(predict.view(-1, args.num_classes), y.view(-1))
                    if proximal_reference_state is not None:
                        prox_penalty = compute_fedprox_penalty(model, proximal_reference_state, args.device)
                        loss = loss + 0.5 * args.fedprox_mu * prox_penalty
                    if not torch.isfinite(loss):
                        event = append_nan_event(
                            local_round_nan_events,
                            stage='loss',
                            epoch=epoch,
                            client_name=cur_single_client,
                            inner_epoch=inner_epoch,
                            step=step,
                            loss=loss.detach().float().item(),
                            lr=optimizer.param_groups[0]['lr'],
                            reason='non-finite loss',
                        )
                        round_non_finite_events += 1
                        if first_non_finite_event is None:
                            first_non_finite_event = event
                            print(f"[NonFinite][Round {epoch}] client={cur_single_client} inner_epoch={inner_epoch} step={step} reason={event['reason']} lr={event['lr']:.6f}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    loss.backward()
                    if has_non_finite_gradients(model):
                        event = append_nan_event(
                            local_round_nan_events,
                            stage='backward',
                            epoch=epoch,
                            client_name=cur_single_client,
                            inner_epoch=inner_epoch,
                            step=step,
                            loss=loss.detach().float().item(),
                            lr=optimizer.param_groups[0]['lr'],
                            reason='non-finite gradients',
                        )
                        round_non_finite_events += 1
                        if first_non_finite_event is None:
                            first_non_finite_event = event
                            print(f"[NonFinite][Round {epoch}] client={cur_single_client} inner_epoch={inner_epoch} step={step} reason={event['reason']} lr={event['lr']:.6f}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    if args.grad_clip:
                        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
                        if total_norm > args.max_grad_norm:
                            round_gradient_stats['clip_count'] += 1
                    else:
                        total_norm = compute_grad_norm(model)
                    if not np.isfinite(total_norm):
                        event = append_nan_event(
                            local_round_nan_events,
                            stage='backward',
                            epoch=epoch,
                            client_name=cur_single_client,
                            inner_epoch=inner_epoch,
                            step=step,
                            loss=loss.detach().float().item(),
                            grad_norm=total_norm,
                            lr=optimizer.param_groups[0]['lr'],
                            reason='non-finite gradient norm',
                        )
                        round_non_finite_events += 1
                        if first_non_finite_event is None:
                            first_non_finite_event = event
                            print(f"[NonFinite][Round {epoch}] client={cur_single_client} inner_epoch={inner_epoch} step={step} reason={event['reason']} lr={event['lr']:.6f}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # 更新梯度统计
                    round_gradient_stats['max_grad_norm'] = max(round_gradient_stats['max_grad_norm'], total_norm)
                    round_gradient_stats['total_grad_norm'] += total_norm
                    round_gradient_stats['step_count'] += 1
                    
                    # 参数更新
                    optimizer.step()
                    
                    args.learning_rate_record[proxy_single_client].append(optimizer.param_groups[0]['lr'])
                    client_last_loss = loss.item()
                    client_last_grad_norm = total_norm
                    client_last_lr = optimizer.param_groups[0]['lr']
                    client_step_count += 1

                    if scheduler is not None and args.decay_type != 'step':
                        scheduler.step()

                if scheduler is not None and args.decay_type == 'step':
                    scheduler.step()

            if client_step_count > 0:
                print(
                    f'Client {cur_single_client} | Rank {getattr(args, "rank", 0)} '
                    f'| Steps: {client_step_count} '
                    f'| Final Loss: {client_last_loss:.4f} '
                    f'| Final GradNorm: {client_last_grad_norm:.4f} '
                    f'| Final LR: {client_last_lr:.6f}'
                )
            else:
                print(
                    f'Client {cur_single_client} | Rank {getattr(args, "rank", 0)} '
                    f'| No optimizer step completed'
                )

            model.to('cpu')  # 释放显存
            maybe_release_cuda_cache(args)
            shutdown_dataloader_workers(train_loader)
            del train_loader
            del optimizer
            del scheduler
            gc.collect()

        if not getattr(args, 'empty_cache_per_client', False):
            torch.cuda.empty_cache()

        round_gradient_stats = reduce_round_gradient_stats(args, round_gradient_stats)
        merged_round_nan_events, round_non_finite_events, first_non_finite_event = collect_round_nan_events(
            args,
            local_round_nan_events,
        )
        if is_main_process(args):
            nan_debug_events.extend(merged_round_nan_events)

        synchronize_local_training_results(args, model_all, local_client_pairs)

        # 打印本轮梯度统计信息
        avg_grad_norm = 0
        clip_rate = 0
        if round_gradient_stats['step_count'] > 0:
            avg_grad_norm = round_gradient_stats['total_grad_norm'] / round_gradient_stats['step_count']
            clip_rate = round_gradient_stats['clip_count'] / round_gradient_stats['step_count'] * 100
            
            print(f"\n=== Round {epoch} Gradient Statistics ===")
            print(f"Max gradient norm: {round_gradient_stats['max_grad_norm']:.4f}")
            print(f"Average gradient norm: {avg_grad_norm:.4f}")
            print(f"Gradient clip rate: {clip_rate:.2f}% ({round_gradient_stats['clip_count']}/{round_gradient_stats['step_count']})")
            
            # 记录到全局统计
            if is_main_process(args):
                gradient_stats['max_grad_norms'].append(round_gradient_stats['max_grad_norm'])
                gradient_stats['avg_grad_norms'].append(avg_grad_norm)
                gradient_stats['grad_clip_count'] += round_gradient_stats['clip_count']
            
        if round_non_finite_events > 0:
            print(f"Round {epoch} skipped {round_non_finite_events} non-finite step(s)")
            if first_non_finite_event is not None:
                print(
                    "First non-finite event: "
                    f"client={first_non_finite_event['client']}, "
                    f"inner_epoch={first_non_finite_event['inner_epoch']}, "
                    f"step={first_non_finite_event['step']}, "
                    f"stage={first_non_finite_event['stage']}, "
                    f"reason={first_non_finite_event['reason']}"
                )

        exp1_branch_vecs = {}
        if exp1_enabled and epoch % args.exp1_interval == 0:
            exp1_record = {'round': epoch}
            for branch in ['left', 'right']:
                client_vecs = [get_branch_params(model_all[client], branch) for client in args.proxy_clients]
                exp1_branch_vecs[branch] = client_vecs
                branch_dim = int(client_vecs[0].numel())
                exp1_record[f'{branch}_param_count'] = branch_dim
                client_vecs_stack = torch.stack(client_vecs)
                exp1_record[f'{branch}_inter_variance'] = client_vecs_stack.var(dim=0, unbiased=False).mean().item()

                pairwise_dists = []
                pairwise_dists_normalized = []
                pairwise_cosine_dists = []
                for i in range(len(client_vecs)):
                    for j in range(i + 1, len(client_vecs)):
                        pairwise_dists.append(compute_l2_distance(client_vecs[i], client_vecs[j]))
                        pairwise_dists_normalized.append(
                            compute_normalized_l2_distance(client_vecs[i], client_vecs[j])
                        )
                        pairwise_cosine_dists.append(
                            compute_cosine_distance(client_vecs[i], client_vecs[j])
                        )
                exp1_record[f'{branch}_pairwise_l2_mean'] = (
                    float(sum(pairwise_dists) / len(pairwise_dists)) if pairwise_dists else 0.0
                )
                exp1_record[f'{branch}_pairwise_l2_normalized_mean'] = (
                    float(sum(pairwise_dists_normalized) / len(pairwise_dists_normalized))
                    if pairwise_dists_normalized else 0.0
                )
                exp1_record[f'{branch}_pairwise_cosine_distance_mean'] = (
                    float(sum(pairwise_cosine_dists) / len(pairwise_cosine_dists))
                    if pairwise_cosine_dists else 0.0
                )
            divergence_log.append(exp1_record)

        if exp2_enabled and epoch % args.exp2_interval == 0:
            cka_record = {'round': epoch}
            client_left_features = []
            client_right_features = []

            try:
                for client in args.proxy_clients:
                    left_feat, right_feat, target_analysis_block_name = extract_branch_features(
                        model_all[client],
                        fixed_test_loader,
                        device=args.device,
                        target_block_name=target_analysis_block_name,
                        img_size=args.img_size,
                    )
                    client_left_features.append(left_feat)
                    client_right_features.append(right_feat)
            except RuntimeError as exc:
                exp2_enabled = False
                print(f"Skip experiment 2 analysis after runtime check failed: {exc}")
            else:
                left_ckas = []
                right_ckas = []
                for i in range(len(client_left_features)):
                    for j in range(i + 1, len(client_left_features)):
                        left_ckas.append(linear_cka(client_left_features[i], client_left_features[j]))
                        right_ckas.append(linear_cka(client_right_features[i], client_right_features[j]))

                cka_record['target_block_name'] = target_analysis_block_name
                cka_record['left_cka_mean'] = float(sum(left_ckas) / len(left_ckas)) if left_ckas else 1.0
                cka_record['left_cka_std'] = torch.tensor(left_ckas).std(unbiased=False).item() if left_ckas else 0.0
                cka_record['right_cka_mean'] = float(sum(right_ckas) / len(right_ckas)) if right_ckas else 1.0
                cka_record['right_cka_std'] = torch.tensor(right_ckas).std(unbiased=False).item() if right_ckas else 0.0
                cka_log.append(cka_record)
                print(
                    f"Round {epoch}: Left CKA={cka_record['left_cka_mean']:.4f}, "
                    f"Right CKA={cka_record['right_cka_mean']:.4f}"
                )

        # 模型聚合
        if is_main_process(args):
            average_model(args, model_avg, model_all)
        broadcast_global_model_state(args, model_avg, model_all)

        if exp1_enabled and exp1_branch_vecs:
            exp1_record = divergence_log[-1]
            for branch in ['left', 'right']:
                exp1_record.setdefault(f'{branch}_param_count', get_branch_param_count(model_avg, branch))
                global_vec = get_branch_params(model_avg, branch)
                reset_dists = [compute_l2_distance(client_vec, global_vec) for client_vec in exp1_branch_vecs[branch]]
                reset_dists_normalized = [
                    compute_normalized_l2_distance(client_vec, global_vec)
                    for client_vec in exp1_branch_vecs[branch]
                ]
                reset_cosine_dists = [
                    compute_cosine_distance(client_vec, global_vec)
                    for client_vec in exp1_branch_vecs[branch]
                ]
                exp1_record[f'{branch}_reset_l2_mean'] = float(sum(reset_dists) / len(reset_dists)) if reset_dists else 0.0
                exp1_record[f'{branch}_reset_l2_normalized_mean'] = (
                    float(sum(reset_dists_normalized) / len(reset_dists_normalized))
                    if reset_dists_normalized else 0.0
                )
                exp1_record[f'{branch}_reset_cosine_distance_mean'] = (
                    float(sum(reset_cosine_dists) / len(reset_cosine_dists))
                    if reset_cosine_dists else 0.0
                )

        # 按频率验证（每 val_freq 轮验证一次，最后一轮也验证）
        is_last_round = (epoch == args.max_communication_rounds - 1)
        should_validate = (epoch % args.val_freq == 0) or is_last_round
        
        if should_validate and is_main_process(args):
            if args.fl_method == 'fedbn':
                validate_fedbn_shared_loader(args, model_avg, client_max_acc, val_loader)
            else:
                # 仅验证聚合后的全局模型一次
                args.single_client = 'global'
                args.best_acc.setdefault('global', float('-inf') if args.num_classes != 1 else float('inf'))
                args.best_eval_loss.setdefault('global', float('inf'))
                args.current_acc.setdefault('global', 0.0)
                args.current_test_acc.setdefault('global', 0.0)
                model_avg.to(args.device)
                valid(args, model_avg, client_max_acc, val_loader, None, TestFlag=False)
                model_avg.cpu()
        else:
            print(f"Skip validation (val_freq={args.val_freq}, next val at round {((epoch // args.val_freq) + 1) * args.val_freq})")

        if is_main_process(args) and client_max_acc:
            max_client = max(client_max_acc.items(), key=lambda x: x[1])  # 找到最大值及其对应的客户端
            max_acc = max_client[1]  # 最大准确率
            max_client_name = max_client[0]  # 对应的客户端名称
            print(f"@@@@@@@@@@@@@@Maximum Accuracy@@@@@@@@@@@@@: {max_acc:.4f}")
            print(f"Achieved by Client: {max_client_name}")
            print(f"[RoundSummary] seed={args.seed} round={epoch} best_acc={max_acc:.4f} best_client={max_client_name}", flush=True)
        elif is_main_process(args):
            max_acc = 0.0
            max_client_name = "N/A"
            print(f"[RoundSummary] seed={args.seed} round={epoch} best_acc={max_acc:.4f} best_client={max_client_name}", flush=True)

        # 记录本轮结果
        round_end_time = datetime.now()
        round_duration = (round_end_time - round_start_time).total_seconds()
        if is_main_process(args):
            print(
                f"Round {epoch} duration: {round_duration:.2f}s "
                f"({round_duration / 60.0:.2f} min)"
            )
            global_validation_accuracy = float(args.current_acc.get('global')) if should_validate else None
            global_test_accuracy = None

            round_result = {
                'round': epoch,
                'validated': bool(should_validate),
                'global_validation_accuracy': global_validation_accuracy,
                'max_accuracy': float(max_acc),
                'best_client': str(max_client_name),
                'gradient_stats': {
                    'max_grad_norm': float(round_gradient_stats['max_grad_norm']),
                    'avg_grad_norm': float(avg_grad_norm),
                    'clip_rate': float(clip_rate)
                },
                'round_duration_seconds': float(round_duration),
                'timestamp': round_end_time.strftime("%Y-%m-%d %H:%M:%S")
            }

            round_results.append(round_result)

        distributed_barrier(args)

        # 终止条件
        epoch += 1

    if is_main_process(args):
        total_training_duration = (datetime.now() - training_start_time).total_seconds()
        avg_round_duration = float(np.mean([item['round_duration_seconds'] for item in round_results])) if round_results else 0.0
        print(f"\nTraining completed after {args.max_communication_rounds} rounds")
        print(f"Total training time: {total_training_duration:.2f}s ({total_training_duration / 60.0:.2f} min)")
        print(f"Average round time: {avg_round_duration:.2f}s ({avg_round_duration / 60.0:.2f} min)")
        print(f"Final Historical Max Accuracy: {max_acc:.4f}")

        print(f"\n=== Final Gradient Statistics ===")
        print(f"Total gradient clips: {gradient_stats['grad_clip_count']}")
        print(f"Total non-finite events skipped: {len(nan_debug_events)}")
        if gradient_stats['max_grad_norms']:
            print(f"Overall max gradient norm: {max(gradient_stats['max_grad_norms']):.4f}")
            print(f"Overall avg gradient norm: {np.mean(gradient_stats['avg_grad_norms']):.4f}")

        final_stats = {
            'final_max_accuracy': float(max_acc),
            'final_best_client': str(max_client_name),
            'total_training_rounds': args.max_communication_rounds,
            'total_gradient_clips': int(gradient_stats['grad_clip_count']),
            'total_non_finite_events': int(len(nan_debug_events)),
            'overall_max_grad_norm': float(max(gradient_stats['max_grad_norms'])) if gradient_stats['max_grad_norms'] else 0.0,
            'overall_avg_grad_norm': float(np.mean(gradient_stats['avg_grad_norms'])) if gradient_stats['avg_grad_norms'] else 0.0,
            'total_training_time_seconds': float(total_training_duration),
            'average_round_time_seconds': float(avg_round_duration),
            'experiment_completed': True,
            'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("================ End Training ================")


def run_worker(local_rank, args):
    try:
        if getattr(args, 'distributed_train', False):
            setup_distributed(args, local_rank)
        else:
            args.local_rank = 0
            args.rank = 0
            args.world_size = 1

        configure_runtime_batch_sizes(args)

        model = initization_configure(args)
        train(args, model)

        if is_main_process(args):
            final_global_acc = args.current_acc.get('global')
            if final_global_acc is None:
                scalar_accs = [val for val in args.current_acc.values() if np.isscalar(val)]
                final_global_acc = float(np.asarray(scalar_accs).mean()) if scalar_accs else 0.0

            message = '\n \n ==============Start showing final performance ================= \n'
            message += 'Final union validation accuracy is: %2.5f  \n' % final_global_acc
            message += "================ End ================ \n"


            print(message)
    finally:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type=str, default="ViT-ev", help="Basic Name of this run with detailed network-architecture selection.")
    parser.add_argument("--FL_platform", type=str, default="ViT-FedAVG", choices=["Swin-FedAVG", "ViT-FedAVG", "Swin-FedAVG", "ResNet-FedAVG"], help="Choose of different FL platform.")
    
    #--------DataSet&&ALPHA------------------------------------------------------------------
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet", "femnist", "femnist-v2", "Retina"], default="cifar10", help="Which dataset.")
    parser.add_argument("--alpha", default=0.1, choices=[0.01, 0.1, 0.5, 5.0], type=float, help="Dirichlet alpha for on-the-fly client partitioning")
    parser.add_argument("--num_clients", default=50, type=int, help="Number of runtime-generated clients for CIFAR/Tiny-ImageNet, or the number of sampled natural FEMNIST clients")
    parser.add_argument("--select_client", default=10, type=int, help="Number of clients selected in each communication round. -1 indicates all clients")
    parser.add_argument("--data_path", type=str, default='./data_total/', help="Where is dataset located.")
    parser.add_argument(
        '--model',
        default='fed-litevit',
        type=str,
        metavar='MODEL',
        choices=[
            'fed-litevit',
            'fed-litevit_Parallel',
            'fed-litevit_LeftOnly',
            'fed-litevit_RightOnly',
            'fed-litevit_RightG1',
            'fed-litevit_NoBranch',
            'fed-litevit_GlobalNoMFE',
            'fed-litevit_Parallel_LeftOnly',
            'fed-litevit_Parallel_RightOnly',
            'FedLiteViT_M1',
            'FedLiteViT_M2',
            'FedLiteViT_M3',
            'FedLiteViT_M4',
            'FedLiteViT_M5',
            'OnDev-LCT-4/1',
            'OnDev-LCT-8/1',
            'CCT-4/2',
            'MobileNetV2',
            'ResNet-32',
            'std-vit-6b',  
            'std-vit-8b',
        ],
        help='Name of model to train',
    )
    parser.add_argument("--save_model_flag", action='store_true', default=False, help="Save the best model for each client.")
    parser.add_argument("--cfg", type=str, default="configs/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to args file for Swin-FL',)

    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")

    parser.add_argument('--Pretrained', action='store_true', default=False, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/swin_tiny_patch4_window7_224.pth", help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="adamw", choices=["sgd", "adam", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--fl_method", default="fedavg", choices=["fedavg", "fedprox", "fedbn"], type=str, help="Federated optimization method. FedProx adds a proximal penalty during local training; FedBN keeps client BatchNorm states local")
    parser.add_argument("--fedprox_mu", default=0.01, type=float, help="FedProx proximal coefficient mu. Only used when fl_method=fedprox")
    parser.add_argument("--num_workers", default=1, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0.025, choices=[0.025, 0], type=float, help="Weight decay if we apply some.")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")
    parser.add_argument("--disable_cifar_tensor_cache", action='store_true', default=False, help="Disable CIFAR tensor cache for the data pipeline")
    parser.add_argument("--empty_cache_per_client", action='store_true', default=False, help="Force torch.cuda.empty_cache() after every client training")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution. fed-litevit family supports 224 and 32.")
    parser.add_argument("--batch_size", default=64, type=int, help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="Comma-separated CUDA device ids, for example 0 or 0,1")
    parser.add_argument("--num_gpus", type=int, default=1, choices=[1, 2], help="Number of GPUs to use from gpu_ids. Set 1 for single-GPU or 2 for dual-GPU client-parallel training")

    parser.add_argument('--seed', type=int, default=None, help="Random seed for initialization. Omit to run seeds 42, 43, and 44 in parallel.")

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear decay.")
    parser.add_argument("--step_size", default=0, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")  # 您可以调整这个值
    parser.add_argument("--lr", dest="learning_rate", default=1e-3, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--enable_lr_scheduler", action='store_true', default=True, help="Enable learning rate scheduling during local training")

    ## FL related parameters
    parser.add_argument("--E_epoch", default=2, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=250, type=int, help="Total communication rounds")
    parser.add_argument("--val_freq", default=1, type=int, help="Validate every N rounds (default: 1)")

    parser.add_argument("--enable_model_complexity", action='store_true', default=False, help="Run model complexity analysis before training")
    parser.add_argument("--clean_output_cache", dest='clean_output_cache', action='store_true', default=True, help="Clean stale caches/output folders before training")
    parser.add_argument("--keep_output_cache", dest='clean_output_cache', action='store_false', help="Keep existing output/cache folders before training")
    parser.add_argument("--run_tag", default="", type=str, help="Optional suffix appended to the experiment directory name to distinguish concurrent runs")
    parser.add_argument("--enable_extra_analysis", action='store_true', default=False, help="Enable exp1 divergence and exp2 CKA analysis during training")
    parser.add_argument("--exp1_interval", default=10, type=int, help="Record divergence/reset analysis every N rounds")
    parser.add_argument("--exp2_interval", default=25, type=int, help="Record CKA analysis every N rounds")
    parser.add_argument("--cka_num_samples", default=1000, type=int, help="Number of fixed validation samples used for CKA analysis")

    # Transform related parameters (used by datasets.py build_transform)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--enable_data_augmentation', action='store_true', default=True, help='Enable training data augmentation')
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT', help='Color jitter factor (default: 0.0)')
    parser.add_argument('--aa', type=str, default='', metavar='NAME', help='AutoAugment policy. Disabled by default')
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=0, help='Random erase count (default: 1)')
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19','MINI'], type=str, help='Image Net dataset path')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    args = parser.parse_args()
    if args.seed is None:
        print(f"No seed provided. Launching parallel runs for seeds: {', '.join(map(str, DEFAULT_PARALLEL_SEEDS))}")
        launch_default_seed_runs(sys.argv[1:], args.output_dir)
        return
    else:
        print(f"Using provided seed: {args.seed}")

    args.distributed_train = args.num_gpus > 1
    if args.distributed_train:
        args.dist_url = f"tcp://127.0.0.1:{find_free_port()}"
        mp.spawn(run_worker, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        run_worker(0, args)


if __name__ == "__main__":
    main()
