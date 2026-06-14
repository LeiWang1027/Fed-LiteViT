#!/home/cherry/miniconda3/envs/fed-lite/bin/python

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.start_config import initization_configure


DEFAULT_DATASETS = ("cifar10", "cifar100", "tinyimagenet", "femnist")
DEFAULT_LRS = {
    "cifar10": 3e-3,
    "cifar100": 5e-3,
    "tinyimagenet": 5e-4,
    "femnist": 1e-3,
    "femnist-v2": 1e-3,
}

RESNET_PLATFORM_MODELS = {"MobileNetV2", "ResNet-32"}


def parse_dataset_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_DATASETS)
    return [item.strip() for item in value.split(",") if item.strip()]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataloader_kwargs(args) -> dict:
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_optimizer(args, model):
    if args.optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    if args.optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )


def concat_arrays(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        raise ValueError("No arrays were provided for centralized training data.")
    return np.concatenate(parts, axis=0)


def build_centralized_train_client(args) -> tuple[int, int]:
    train_data_parts = []
    train_label_parts = []
    client_names = [
        name
        for name in args.dis_cvs_files
        if name in args.client_data and "train" in args.client_data[name]
    ]

    for client_name in client_names:
        train_split = args.client_data[client_name]["train"]
        train_data_parts.append(np.asarray(train_split["data"]))
        train_label_parts.append(np.asarray(train_split["labels"]))

    centralized_name = "centralized_train"
    args.client_data[centralized_name] = {
        "train": {
            "data": concat_arrays(train_data_parts),
            "labels": concat_arrays(train_label_parts),
        }
    }
    args.single_client = centralized_name
    args.dis_cvs_files = [centralized_name]
    args.clients_with_len = {
        centralized_name: len(args.client_data[centralized_name]["train"]["labels"])
    }
    args.best_acc = {centralized_name: 0.0}
    args.current_acc = {}
    args.current_test_acc = {}
    args.best_eval_loss = {centralized_name: float("inf")}
    return len(client_names), args.clients_with_len[centralized_name]


def maybe_resize_inputs(inputs: torch.Tensor, img_size: int) -> torch.Tensor:
    if inputs.shape[-1] == img_size and inputs.shape[-2] == img_size:
        return inputs
    return F.interpolate(inputs, size=(img_size, img_size), mode="bilinear", align_corners=False)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    return correct, targets.numel()


def evaluate(args, model, data_loader, split_name: str) -> dict:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0
    sample_sum = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(data_loader, start=1):
            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            inputs = maybe_resize_inputs(inputs, args.img_size)
            logits = model(inputs)
            loss = criterion(logits, targets)
            correct, count = accuracy_from_logits(logits, targets)
            loss_sum += loss.item() * count
            correct_sum += correct
            sample_sum += count
            if args.max_eval_batches > 0 and batch_index >= args.max_eval_batches:
                break

    model.train()
    return {
        f"{split_name}_loss": loss_sum / max(sample_sum, 1),
        f"{split_name}_acc": correct_sum / max(sample_sum, 1),
        f"{split_name}_samples": sample_sum,
    }



def clone_model_state_cpu(model) -> dict:
    model_to_save = model.module if hasattr(model, "module") else model
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model_to_save.state_dict().items()
    }


def restore_model_state(model, state: dict, device) -> None:
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(
        {name: tensor.to(device=device) for name, tensor in state.items()},
        strict=True,
    )


def model_has_non_finite_tensors(model) -> bool:
    model_to_check = model.module if hasattr(model, "module") else model
    for tensor in model_to_check.state_dict().values():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            return True
    return False


def gradients_are_finite(model) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True



def configure_model_platform(args) -> None:
    if args.model in RESNET_PLATFORM_MODELS:
        args.FL_platform = "ResNet-Centralized"


def build_run_args(base_args, dataset: str, seed: int):
    args = deepcopy(base_args)
    args.dataset = dataset
    args.seed = seed
    args.learning_rate = (
        args.dataset_lrs.get(dataset, DEFAULT_LRS.get(dataset, 1e-3))
        if args.learning_rate_override is None
        else args.learning_rate_override
    )
    args.net_name = args.net_name or f"ViT-ev-fed-litevit_centralized_{dataset}"
    args.run_tag = args.run_tag or "centralized"
    args.input_size = args.img_size
    args.data_set = "MINI" if dataset == "tinyimagenet" else "CIFAR"
    args.distributed_train = False
    args.num_gpus = 1
    args.fl_method = "centralized"
    configure_model_platform(args)
    if args.max_communication_rounds <= 0:
        args.max_communication_rounds = args.epochs
    args.select_client = args.num_clients
    args.enable_cifar_tensor_cache = not args.disable_cifar_tensor_cache
    return args


def run_one_dataset(base_args, dataset: str, seed: int) -> dict:
    args = build_run_args(base_args, dataset, seed)
    set_random_seed(seed)

    model = initization_configure(args)
    create_dataset_and_evalmetrix(args)
    source_clients, train_samples = build_centralized_train_client(args)

    loader_kwargs = dataloader_kwargs(args)
    train_set = DatasetFLViT(args, "train")
    val_set = DatasetFLViT(args, "val")
    test_set = DatasetFLViT(args, "test")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    optimizer = build_optimizer(args, model)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs * len(train_loader), 1),
        )
        if args.enable_lr_scheduler
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    nonfinite_events = []
    skipped_nonfinite_batches = 0
    best_val_acc = -1.0
    best_epoch = -1
    start_time = time.time()
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    last_good_state = clone_model_state_cpu(model)

    print(
        f"[Centralized] dataset={dataset} seed={seed} "
        f"source_clients={source_clients} train_samples={train_samples} "
        f"val_samples={len(val_set)} test_samples={len(test_set)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct_sum = 0
        epoch_sample_sum = 0
        epoch_skipped_nonfinite = 0
        epoch_start_time = time.time()

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            inputs = maybe_resize_inputs(inputs, args.img_size)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(inputs)
                loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                skipped_nonfinite_batches += 1
                epoch_skipped_nonfinite += 1
                nonfinite_events.append({
                    "dataset": dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "step": step,
                    "reason": "loss",
                    "loss": str(loss.detach().item()),
                    "lr": optimizer.param_groups[0]["lr"],
                })
                restore_model_state(model, last_good_state, args.device)
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[NonFiniteRecovery] dataset={dataset} epoch={epoch} step={step} "
                    "loss is non-finite; restored last good model state and skipped batch."
                )
                if skipped_nonfinite_batches > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"Exceeded --max_nonfinite_batches={args.max_nonfinite_batches} "
                        f"for dataset={dataset}."
                    )
                continue

            scaler.scale(loss).backward()
            if args.amp:
                scaler.unscale_(optimizer)
            if args.grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=False,
                )
            else:
                grad_norm = torch.zeros((), device=args.device)

            if not gradients_are_finite(model) or not torch.isfinite(grad_norm):
                skipped_nonfinite_batches += 1
                epoch_skipped_nonfinite += 1
                nonfinite_events.append({
                    "dataset": dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "step": step,
                    "reason": "gradient",
                    "loss": float(loss.detach().item()),
                    "grad_norm": str(grad_norm.detach().item()),
                    "lr": optimizer.param_groups[0]["lr"],
                })
                restore_model_state(model, last_good_state, args.device)
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[NonFiniteRecovery] dataset={dataset} epoch={epoch} step={step} "
                    "gradient is non-finite; restored last good model state and skipped batch."
                )
                if skipped_nonfinite_batches > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"Exceeded --max_nonfinite_batches={args.max_nonfinite_batches} "
                        f"for dataset={dataset}."
                    )
                continue

            scaler.step(optimizer)
            scaler.update()
            if model_has_non_finite_tensors(model):
                skipped_nonfinite_batches += 1
                epoch_skipped_nonfinite += 1
                nonfinite_events.append({
                    "dataset": dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "step": step,
                    "reason": "parameter",
                    "loss": float(loss.detach().item()),
                    "grad_norm": str(grad_norm.detach().item()),
                    "lr": optimizer.param_groups[0]["lr"],
                })
                restore_model_state(model, last_good_state, args.device)
                optimizer = build_optimizer(args, model)
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
                scheduler = None
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[NonFiniteRecovery] dataset={dataset} epoch={epoch} step={step} "
                    "parameter became non-finite; restored last good model state and reset optimizer."
                )
                if skipped_nonfinite_batches > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"Exceeded --max_nonfinite_batches={args.max_nonfinite_batches} "
                        f"for dataset={dataset}."
                    )
                continue

            last_good_state = clone_model_state_cpu(model)
            if scheduler is not None:
                scheduler.step()

            correct, count = accuracy_from_logits(logits.detach(), targets)
            epoch_loss_sum += loss.item() * count
            epoch_correct_sum += correct
            epoch_sample_sum += count

            if args.log_interval > 0 and step % args.log_interval == 0:
                print(
                    f"[Train] dataset={dataset} epoch={epoch}/{args.epochs} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item():.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g}"
                )
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break

        train_metrics = {
            "train_loss": epoch_loss_sum / max(epoch_sample_sum, 1),
            "train_acc": epoch_correct_sum / max(epoch_sample_sum, 1),
            "train_samples": epoch_sample_sum,
        }

        row = {
            "dataset": dataset,
            "seed": seed,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_start_time, 3),
            "skipped_nonfinite_batches": epoch_skipped_nonfinite,
            **train_metrics,
        }

        if epoch % args.val_freq == 0 or epoch == args.epochs:
            row.update(evaluate(args, model, val_loader, "val"))
            row.update(evaluate(args, model, test_loader, "test"))
            if row["val_acc"] > best_val_acc:
                best_val_acc = row["val_acc"]
                best_epoch = epoch
                if args.save_model_flag:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "metrics": row,
                        },
                        best_checkpoint_path,
                    )

        history.append(row)
        print(
            f"[EpochSummary] dataset={dataset} seed={seed} epoch={epoch} "
            f"train_acc={row['train_acc']:.4f} "
            f"val_acc={row.get('val_acc', float('nan')):.4f} "
            f"test_acc={row.get('test_acc', float('nan')):.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )

    final_row = history[-1]
    result = {
        "dataset": dataset,
        "seed": seed,
        "model": args.model,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer_type": args.optimizer_type,
        "source_clients": source_clients,
        "train_samples": train_samples,
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_acc": final_row.get("train_acc"),
        "final_val_acc": final_row.get("val_acc"),
        "final_test_acc": final_row.get("test_acc"),
        "skipped_nonfinite_batches": skipped_nonfinite_batches,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "output_dir": args.output_dir,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }

    return result


def parse_dataset_lrs(values: list[str]) -> dict[str, float]:
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-lr entry: {item}. Expected dataset=lr.")
        dataset, lr = item.split("=", 1)
        result[dataset.strip()] = float(lr)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Fed-LiteViT in a centralized training setting across datasets."
    )
    parser.add_argument("--datasets", default="cifar10", help="Comma-separated datasets or 'all'.")
    parser.add_argument("--model", default="fed-litevit", type=str)
    parser.add_argument("--net_name", default="", type=str)
    parser.add_argument("--FL_platform", default="ViT-Centralized", type=str)
    parser.add_argument("--output_dir", default="output-centralized", type=str)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--num_clients", default=100, type=int, help="Source split count used before merging into centralized data.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Source Dirichlet alpha used before merging CIFAR/Tiny-ImageNet clients.")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", dest="learning_rate_override", default=None, type=float)
    parser.add_argument("--dataset-lr", action="append", default=[], help="Override per dataset, e.g. --dataset-lr cifar10=0.003")
    parser.add_argument("--optimizer_type", default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--weight_decay", default=0.025, type=float)
    parser.add_argument("--val_freq", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--grad_clip", action="store_true", default=True)
    parser.add_argument("--disable_grad_clip", dest="grad_clip", action="store_false")
    parser.add_argument("--enable_lr_scheduler", action="store_true", default=True)
    parser.add_argument("--disable_lr_scheduler", dest="enable_lr_scheduler", action="store_false")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--save_model_flag", action="store_true", default=False)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--max_train_batches", default=0, type=int, help="Debug limit. 0 means use all train batches.")
    parser.add_argument("--max_eval_batches", default=0, type=int, help="Debug limit. 0 means use all eval batches.")
    parser.add_argument("--max_nonfinite_batches", default=100, type=int, help="Stop only after this many recovered non-finite batches.")
    parser.add_argument("--run_tag", default="centralized", type=str)
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")

    parser.add_argument("--cfg", default="configs/swin_tiny_patch4_window7_224.yaml", type=str)
    parser.add_argument("--distillation-type", dest="distillation_type", default="none", choices=["none", "soft", "hard"])
    parser.add_argument("--Pretrained", action="store_true", default=False)
    parser.add_argument("--pretrained_dir", default="checkpoint/swin_tiny_patch4_window7_224.pth", type=str)
    parser.add_argument("--disable_cifar_tensor_cache", action="store_true", default=False)
    parser.add_argument("--enable_data_augmentation", action="store_true", default=True)
    parser.add_argument("--color-jitter", dest="color_jitter", default=0.0, type=float)
    parser.add_argument("--aa", default="", type=str)
    parser.add_argument("--reprob", default=0.0, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=0, type=int)
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--max_communication_rounds", default=0, type=int)
    parser.add_argument("--E_epoch", default=1, type=int)
    parser.add_argument("--fedprox_mu", default=0.0, type=float)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.dataset_lrs = parse_dataset_lrs(args.dataset_lr)
    datasets = parse_dataset_list(args.datasets)

    if args.dry_run:
        for dataset in datasets:
            run_args = build_run_args(args, dataset, args.seed)
            print(
                " ".join(
                    [
                        "planned:",
                        f"dataset={dataset}",
                        f"model={run_args.model}",
                        f"FL_platform={run_args.FL_platform}",
                        f"img_size={run_args.img_size}",
                        f"epochs={run_args.epochs}",
                        f"batch_size={run_args.batch_size}",
                        f"lr={run_args.learning_rate}",
                        f"output_dir={run_args.output_dir}",
                    ]
                )
            )
        return

    results = []
    for dataset in datasets:
        results.append(run_one_dataset(args, dataset, args.seed))

    print("\n========== Centralized Fed-LiteViT Summary ==========")
    for result in results:
        print(
            f"{result['dataset']}: best_val_acc={result['best_val_acc']:.4f} "
            f"final_test_acc={result.get('final_test_acc', float('nan')):.4f} "
            f"output_dir={result['output_dir']}"
        )
    print("=====================================================")


if __name__ == "__main__":
    main()
