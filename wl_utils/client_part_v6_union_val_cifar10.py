import numpy as np
import os
from torchvision.datasets import CIFAR10 as cifar
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="参数设置")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT', help='Color jitter factor (default: 0.0)')
    parser.add_argument('--aa', type=str, default='', metavar='NAME', help='AutoAugment policy. Disabled by default')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19','MINI'], type=str, help='Image Net dataset path')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    args = parser.parse_args([])  # No arguments passed by default
    return args

args = get_args()

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=InterpolationMode.BILINEAR,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                              interpolation=InterpolationMode.BILINEAR)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def split_data_dirichlet(data, labels, num_clients, alpha, min_samples_per_client=1):
    """
    使用狄利克雷分布将数据集划分给多个客户端，确保每个客户端至少有一些样本。
    """
    num_classes = len(np.unique(labels))
    class_idx = [np.where(labels == i)[0] for i in range(num_classes)]
    per_client_per_class_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    client_idx = [[] for _ in range(num_clients)]
    client_class_counts = {f'client_{i}': np.zeros(num_classes, dtype=int) for i in range(num_clients)}

    for class_id, (indices, distribution) in enumerate(zip(class_idx, per_client_per_class_distribution)):
        np.random.shuffle(indices)
        # Step 1: Ensure each client gets a minimum number of samples
        min_total_samples = min_samples_per_client * num_clients
        remaining_samples = len(indices) - min_total_samples
        if remaining_samples < 0:
            raise ValueError(f"Not enough samples in class {class_id} to allocate {min_samples_per_client} per client.")
        
        # Allocate the minimum number of samples first
        for client_id in range(num_clients):
            client_samples = indices[:min_samples_per_client]
            client_idx[client_id].extend(client_samples)
            client_class_counts[f'client_{client_id}'][class_id] += len(client_samples)
            indices = indices[min_samples_per_client:]

        # Step 2: Distribute remaining samples according to Dirichlet distribution
        remaining_distribution = np.random.dirichlet([alpha] * num_clients)
        samples_per_client = [int(remaining_samples * ratio) for ratio in remaining_distribution]
        total_assigned_samples = sum(samples_per_client)
        samples_to_distribute = remaining_samples - total_assigned_samples
        for _ in range(samples_to_distribute):
            selected_client = np.random.choice(num_clients)
            samples_per_client[selected_client] += 1

        current_pos = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            selected_indices = indices[current_pos:current_pos + num_samples]
            client_idx[client_id].extend(selected_indices)
            client_class_counts[f'client_{client_id}'][class_id] += len(selected_indices)
            current_pos += num_samples

    return client_idx, client_class_counts


def create_clients_data(args, data_path='./data', num_clients=1, alpha=100, save_path='./data'):
    """
    创建并分配数据给各个客户端，每个客户端都有自己的训练集和测试集。
    """
    transform = build_transform(is_train=True, args=args)
    cifar10_train = cifar(data_path, download=True, train=True, transform=transform)
    cifar10_test = cifar(data_path, download=True, train=False, transform=transform)

    # 获取训练和测试数据并混合
    data = np.concatenate([cifar10_train.data, cifar10_test.data], axis=0)
    labels = np.concatenate([np.array(cifar10_train.targets), np.array(cifar10_test.targets)], axis=0)

    # 创建客户端数据字典
    client_data = {f'client_{i}': {} for i in range(num_clients)}

    # 将数据按狄利克雷分布分配给各个客户端
    client_indices, client_class_counts = split_data_dirichlet(data, labels, num_clients, alpha)

    for i, indices in enumerate(client_indices):
        client_data[f'client_{i}']['train'] = {}
        client_data[f'client_{i}']['test'] = {}

        # 将每个客户端的数据按 5:1 划分为训练集和测试集
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.83)
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        client_data[f'client_{i}']['train']['data'] = data[train_indices]
        client_data[f'client_{i}']['train']['labels'] = labels[train_indices]
        client_data[f'client_{i}']['test']['data'] = data[test_indices]
        client_data[f'client_{i}']['test']['labels'] = labels[test_indices]

    # 创建保存目录并保存 client_data
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'client_data.npy'), client_data)

    return client_data, client_class_counts

def count_labels_per_class(data, labels, num_classes):
    """
    统计每个数据集中每个类别的数量。
    """
    counts = np.zeros(num_classes, dtype=int)
    for label in labels:
        counts[label] += 1
    return counts

# 调用函数
client_data, client_class_counts = create_clients_data(args, num_clients=100, alpha=0.1)

# CIFAR-10 数据集有 10 个类别
num_classes = 10

# 输出各客户端训练集和测试集中每个类别的数量
for client, data in client_data.items():
    train_labels = data['train']['labels']
    test_labels = data['test']['labels']

    train_class_counts = count_labels_per_class(data['train']['data'], train_labels, num_classes)
    test_class_counts = count_labels_per_class(data['test']['data'], test_labels, num_classes)

    print(f"{client} 训练集各类别的数据量: {train_class_counts}")
    print(f"{client} 测试集各类别的数据量: {test_class_counts}")

# 输出每个客户端的训练集和测试集大小
for client, data in client_data.items():
    print(f"{client} 训练集大小: {len(data['train']['data'])}, 测试集大小: {len(data['test']['data'])}")

# 输出所有客户端的测试集总量
total_test_data = sum(len(data['test']['data']) for data in client_data.values())
print(f"所有客户端测试集数据总量: {total_test_data}")


total_train_data = sum(len(data['train']['data']) for data in client_data.values())
print(f"所有客户端训练集数据总量: {total_train_data}")