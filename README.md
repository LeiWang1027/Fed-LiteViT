# Fed-LiteViT

Official implementation for the manuscript:

**Fed-LiteViT: A Robust Lightweight Vision Transformer Backbone for Non-IID Edge Federated Visual Learning**

Fed-LiteViT is a lightweight Vision Transformer backbone for robust, low-cost edge federated visual learning under non-IID uncertainty. The core model embeds heterogeneity adaptation into the client architecture through MICLA, a capacity-asymmetric dual-branch mixer that stabilizes client updates while preserving visual representation capacity. This repository provides the code used for the Applied Soft Computing submission, including federated training, centralized sanity checks, model-complexity profiling, and diagnostic analyses.

## Highlights

- Robust lightweight ViT backbone for non-IID edge federated visual learning.
- MICLA uses capacity-asymmetric modulation to stabilize client updates.
- Stage-wise local-global-local attention balances local detail and global calibration.
- Support for matched FL protocols including FedAvg, FedProx, and FedBN.
- Diagnostic utilities cover model complexity, branch divergence, CKA, and ablations.

## Repository Layout

```text
.
|-- fed_litevit/
|   `-- classification/
|       |-- data/                  # Dataset helpers for centralized/classification utilities
|       `-- model/                 # Fed-LiteViT, MICLA, Std-ViT, and related backbones
|-- models/                        # Baseline model builders: CCT, OnDev-LCT, Swin, ResNet
|-- utils/                         # FL data partitioning, training utilities, schedulers
|-- wl_utils/                      # Client partitioning helpers
|-- configs/                       # Swin configuration files
|-- analysis_utils.py              # Divergence and CKA analysis helpers
|-- run_centralized_fed_litevit.py # Centralized training/evaluation entry point
|-- train_FedAVG_share_noaug_nolr_v6_record_macs.py
|                                  # Main federated training entry point
`-- requirements.txt
```

## Main Models

The main training script accepts the following model names through `--model`:

- Fed-LiteViT family: `fed-litevit`, `fed-litevit_Parallel`, `fed-litevit_LeftOnly`, `fed-litevit_RightOnly`, `fed-litevit_RightG1`, `fed-litevit_NoBranch`, `fed-litevit_GlobalNoMFE`, `fed-litevit_Parallel_LeftOnly`, `fed-litevit_Parallel_RightOnly`
- Larger/internal Fed-LiteViT variants: `FedLiteViT_M1`, `FedLiteViT_M2`, `FedLiteViT_M3`, `FedLiteViT_M4`, `FedLiteViT_M5`
- Baselines: `OnDev-LCT-4/1`, `OnDev-LCT-8/1`, `CCT-4/2`, `MobileNetV2`, `ResNet-32`, `std-vit-6b`, `std-vit-8b`

The paper's compact Fed-LiteViT instantiation uses a three-stage pyramid with depths `{1, 1, 1}` and channel widths `{64, 128, 192}`.

## Environment

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The code is based on PyTorch and expects CUDA for full experiments. The current runtime configuration contains a hardware guard in `utils/start_config.py` that checks for an NVIDIA RTX 5080. If you run the code on another CUDA GPU, adjust `_require_rtx_5080` in that file before launching experiments.

## Data

The code supports the public benchmarks used in the paper:

- CIFAR-10
- CIFAR-100
- FEMNIST
- Tiny-ImageNet

Use `--data_path` to specify the dataset root. CIFAR datasets are loaded through `torchvision` and can be downloaded automatically. Tiny-ImageNet and FEMNIST should be placed under the data root in the expected extracted or archive format:

```text
data/
|-- tiny-imagenet-200/             # or tiny-imagenet-200.zip
`-- femnist/                       # or femnist.7z
```

Client data are generated at runtime. For CIFAR and Tiny-ImageNet, client partitions use Dirichlet label skew controlled by `--alpha`. For FEMNIST, clients follow the natural writer-partitioned setting.

## Federated Training

Main entry point:

```bash
python train_FedAVG_share_noaug_nolr_v6_record_macs.py [options]
```

Important options:

- `--dataset`: `cifar10`, `cifar100`, `tinyimagenet`, `femnist`, or `femnist-v2`
- `--model`: backbone name, e.g. `fed-litevit`
- `--fl_method`: `fedavg`, `fedprox`, or `fedbn`
- `--alpha`: Dirichlet non-IID concentration for CIFAR/Tiny-ImageNet
- `--num_clients`: total number of generated/sampled clients
- `--select_client`: number of clients sampled per communication round; `-1` selects all clients
- `--E_epoch`: local epochs per round
- `--max_communication_rounds`: total communication rounds
- `--enable_model_complexity`: print parameter/MAC/FLOP statistics before training
- `--enable_extra_analysis`: enable divergence and CKA diagnostics during training

### Example: Fed-LiteViT with FedAvg on CIFAR-100

```bash
python train_FedAVG_share_noaug_nolr_v6_record_macs.py \
  --FL_platform ViT-FedAVG \
  --model fed-litevit \
  --net_name Fed-LiteViT \
  --dataset cifar100 \
  --data_path ./data \
  --alpha 0.1 \
  --num_clients 50 \
  --select_client 10 \
  --fl_method fedavg \
  --E_epoch 2 \
  --max_communication_rounds 250 \
  --batch_size 64 \
  --img_size 224 \
  --seed 42 \
  --enable_model_complexity
```

### Example: FedProx

```bash
python train_FedAVG_share_noaug_nolr_v6_record_macs.py \
  --model fed-litevit \
  --dataset cifar100 \
  --data_path ./data \
  --alpha 0.1 \
  --num_clients 50 \
  --select_client 10 \
  --fl_method fedprox \
  --fedprox_mu 0.01 \
  --E_epoch 2 \
  --max_communication_rounds 250 \
  --seed 42
```

### Example: FedBN

```bash
python train_FedAVG_share_noaug_nolr_v6_record_macs.py \
  --model fed-litevit \
  --dataset cifar100 \
  --data_path ./data \
  --alpha 0.1 \
  --num_clients 50 \
  --select_client 10 \
  --fl_method fedbn \
  --E_epoch 2 \
  --max_communication_rounds 250 \
  --seed 42
```

### Example: Mechanism Diagnostics

```bash
python train_FedAVG_share_noaug_nolr_v6_record_macs.py \
  --model fed-litevit \
  --dataset cifar10 \
  --data_path ./data \
  --alpha 0.1 \
  --num_clients 50 \
  --select_client 10 \
  --fl_method fedavg \
  --E_epoch 2 \
  --max_communication_rounds 100 \
  --seed 42 \
  --enable_extra_analysis \
  --exp1_interval 10 \
  --exp2_interval 25 \
  --cka_num_samples 1000
```

If `--seed` is omitted in the federated script, the code launches the default seed set `42, 43, 44` in parallel.

## Centralized Training

For centralized sanity checks:

```bash
python run_centralized_fed_litevit.py \
  --datasets cifar10 \
  --model fed-litevit \
  --data_path ./data \
  --epochs 200 \
  --batch_size 128 \
  --img_size 224 \
  --seed 42
```

Use `--datasets all` to run CIFAR-10, CIFAR-100, Tiny-ImageNet, and FEMNIST sequentially.

## Outputs

By default, federated runs write logs, checkpoints, and analysis outputs under `output/`. Centralized runs write under `output-centralized/`. These directories can become large and should normally be excluded from Git.

Recommended metadata to record for each experiment:

- Dataset and data root
- Model name
- FL method
- Dirichlet alpha or FEMNIST client sampling configuration
- Number of clients and selected clients per round
- Local epochs and communication rounds
- Batch size, learning rate, optimizer, and weight decay
- Seed
- GPU type and CUDA/PyTorch versions

## Notes for Reproducibility

- CIFAR and Tiny-ImageNet experiments use synthetic Dirichlet label-skew partitions.
- FEMNIST uses natural writer-partitioned heterogeneity and is treated as complementary evidence rather than a controlled feature-shift benchmark.
- The communication estimate reported in the paper assumes FP32 model download plus upload per communication round and does not include optimizer states.
- Some baseline components are adapted from public implementations of Swin Transformer, CCT, OnDev-LCT, and ResNet-style CIFAR backbones. Please keep the corresponding license notices when redistributing the code.

## Citation

If you use this repository, please cite the paper after publication:

```text
Fed-LiteViT: A Robust Lightweight Vision Transformer Backbone for Non-IID Edge Federated Visual Learning.
Submitted to Applied Soft Computing.
```
