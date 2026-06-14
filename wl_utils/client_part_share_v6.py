import os
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


def split_data_dirichlet(labels, num_clients, alpha, min_samples_per_class=5):
    num_classes = len(np.unique(labels))
    class_idx = [np.where(labels == i)[0] for i in range(num_classes)]
    distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]
    client_class_counts = {
        f"client_{i}": np.zeros(num_classes, dtype=int) for i in range(num_clients)
    }

    for class_id in range(num_classes):
        np.random.shuffle(class_idx[class_id])

        min_samples = min(min_samples_per_class, len(class_idx[class_id]) // num_clients)
        if min_samples > 0:
            for client_id in range(num_clients):
                selected = class_idx[class_id][:min_samples]
                client_indices[client_id].extend(selected)
                client_class_counts[f"client_{client_id}"][class_id] += min_samples
                class_idx[class_id] = class_idx[class_id][min_samples:]

        remaining = len(class_idx[class_id])
        if remaining == 0:
            continue

        proportions = distribution[class_id] / distribution[class_id].sum()
        allocations = (proportions * remaining).astype(int)
        allocations[: remaining - allocations.sum()] += 1

        ptr = 0
        for client_id in range(num_clients):
            alloc = allocations[client_id]
            if alloc == 0:
                continue
            selected = class_idx[class_id][ptr : ptr + alloc]
            client_indices[client_id].extend(selected)
            client_class_counts[f"client_{client_id}"][class_id] += alloc
            ptr += alloc

    return client_indices, client_class_counts


def build_client_data(dataset_name="cifar100", data_path="./data", num_clients=50, alpha=5, min_samples_per_class=5):
    dataset_map = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset for client partitioning: {dataset_name}")

    cifar_dataset = dataset_map[dataset_name]
    train_set = cifar_dataset(data_path, train=True, download=True)
    test_set = cifar_dataset(data_path, train=False, download=True)

    train_data = np.array(train_set.data)
    train_labels = np.array(train_set.targets)

    client_indices, client_class_counts = split_data_dirichlet(
        labels=train_labels,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_class=min_samples_per_class,
    )

    client_data = {}
    for client_id in range(num_clients):
        indices = np.array(client_indices[client_id], dtype=int)
        np.random.shuffle(indices)
        client_data[f"client_{client_id}"] = {
            "train": {
                "data": train_data[indices],
                "labels": train_labels[indices],
            }
        }

    shared_test = {
        "data": np.array(test_set.data),
        "labels": np.array(test_set.targets),
    }
    client_data["shared_val_set"] = shared_test
    client_data["shared_test_set"] = shared_test

    total_allocated = sum(
        len(v["train"]["data"]) for k, v in client_data.items() if k.startswith("client_")
    )
    assert total_allocated == len(train_data), (
        f"allocated samples {total_allocated} do not match training set size {len(train_data)}"
    )

    return client_data, client_class_counts


def create_clients_data(dataset_name="cifar100", data_path="./data", num_clients=50, alpha=5, save_path="./clients", min_samples_per_class=5):
    client_data, client_class_counts = build_client_data(
        dataset_name=dataset_name,
        data_path=data_path,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_class=min_samples_per_class,
    )

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "client_data.npy"), client_data)
    return client_data, client_class_counts


if __name__ == "__main__":
    create_clients_data(dataset_name="cifar100", num_clients=10, alpha=0.1, save_path="./data")
