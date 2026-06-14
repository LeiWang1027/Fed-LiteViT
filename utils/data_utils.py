import os
import random
import json
import shutil
import subprocess
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from fed_litevit.classification.data.datasets import build_transform

import torch
from torchvision import transforms
import torch.utils.data as data
from wl_utils.client_part_share_v6 import build_client_data, split_data_dirichlet

Image.LOAD_TRUNCATED_IMAGES = True


_CIFAR_UINT8_CACHE = {}
_CIFAR_EVAL_CACHE = {}
_FEMNIST_CLASS_COUNT = 62
_TINYIMAGENET_CLASS_COUNT = 200


def _resolve_data_root(args, extracted_dir_name, archive_names):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    search_roots = []
    configured_root = getattr(args, "data_path", None)
    if configured_root:
        if os.path.isabs(configured_root):
            search_roots.append(configured_root)
        else:
            search_roots.append(os.path.abspath(configured_root))
            search_roots.append(os.path.abspath(os.path.join(repo_root, configured_root)))
    search_roots.append(os.path.abspath(os.path.join(repo_root, "data")))
    search_roots.append(os.path.abspath("./data"))

    seen_roots = set()
    for root in search_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)

        candidate_dirs = [root]
        if os.path.basename(root.rstrip(os.sep)) != extracted_dir_name:
            candidate_dirs.append(os.path.join(root, extracted_dir_name))

        for candidate_dir in candidate_dirs:
            if os.path.isdir(candidate_dir):
                return root

        for archive_name in archive_names:
            candidate_archives = [root]
            if os.path.basename(root.rstrip(os.sep)) != archive_name:
                candidate_archives.append(os.path.join(root, archive_name))
            for candidate_archive in candidate_archives:
                if os.path.exists(candidate_archive):
                    return root

    return os.path.abspath("./data")


def _resolve_dataset_paths(root, extracted_dir_name, archive_names):
    dataset_dir = root
    if os.path.basename(root.rstrip(os.sep)) != extracted_dir_name:
        dataset_dir = os.path.join(root, extracted_dir_name)

    archive_path = None
    for archive_name in archive_names:
        candidate_path = root
        if os.path.basename(root.rstrip(os.sep)) != archive_name:
            candidate_path = os.path.join(root, archive_name)
        if os.path.exists(candidate_path):
            archive_path = candidate_path
            break

    return dataset_dir, archive_path


def _resolve_femnist_data_root(args):
    return _resolve_data_root(args, extracted_dir_name="femnist", archive_names=["femnist.7z"])


def _ensure_femnist_extracted(data_root):
    all_data_dir = os.path.join(data_root, "femnist", "data", "all_data")
    if os.path.isdir(all_data_dir):
        return all_data_dir

    archive_path = os.path.join(data_root, "femnist.7z")
    if not os.path.exists(archive_path):
        raise FileNotFoundError(
            f"FEMNIST archive not found. Expected either {all_data_dir} or {archive_path}."
        )

    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7z is required to extract femnist.7z but was not found in PATH.")

    subprocess.run(
        [seven_zip, "x", "-y", archive_path, f"-o{data_root}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not os.path.isdir(all_data_dir):
        raise FileNotFoundError(f"FEMNIST extraction completed but {all_data_dir} was not found.")

    return all_data_dir


def _resolve_tinyimagenet_data_root(args):
    return _resolve_data_root(
        args,
        extracted_dir_name="tiny-imagenet-200",
        archive_names=["tiny-imagenet-200.zip"],
    )


def _ensure_tinyimagenet_extracted(data_root):
    dataset_dir, archive_path = _resolve_dataset_paths(
        data_root,
        extracted_dir_name="tiny-imagenet-200",
        archive_names=["tiny-imagenet-200.zip"],
    )
    if os.path.isdir(dataset_dir):
        return dataset_dir

    if archive_path is None:
        raise FileNotFoundError(
            f"Tiny-ImageNet archive not found. Expected either {dataset_dir} or tiny-imagenet-200.zip under {data_root}."
        )

    extract_root = data_root if os.path.isdir(data_root) else os.path.dirname(archive_path)
    os.makedirs(extract_root, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_root)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Tiny-ImageNet extraction completed but {dataset_dir} was not found.")

    return dataset_dir


def _load_tinyimagenet_metadata(dataset_dir):
    wnids_path = os.path.join(dataset_dir, "wnids.txt")
    if not os.path.exists(wnids_path):
        raise FileNotFoundError(f"Tiny-ImageNet metadata file not found: {wnids_path}")

    with open(wnids_path, "r", encoding="utf-8") as handle:
        class_names = [line.strip() for line in handle if line.strip()]

    if len(class_names) != _TINYIMAGENET_CLASS_COUNT:
        raise ValueError(
            f"Expected {_TINYIMAGENET_CLASS_COUNT} Tiny-ImageNet classes, found {len(class_names)} in {wnids_path}."
        )

    return {class_name: class_index for class_index, class_name in enumerate(class_names)}


def _load_tinyimagenet_train_split(dataset_dir, class_to_index):
    train_dir = os.path.join(dataset_dir, "train")
    image_paths = []
    labels = []

    for class_name, class_index in class_to_index.items():
        images_dir = os.path.join(train_dir, class_name, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Tiny-ImageNet train images directory not found: {images_dir}")

        for image_name in sorted(os.listdir(images_dir)):
            if image_name.lower().endswith((".jpeg", ".jpg", ".png")):
                image_paths.append(os.path.join(images_dir, image_name))
                labels.append(class_index)

    return np.asarray(image_paths, dtype=object), np.asarray(labels, dtype=np.int64)


def _load_tinyimagenet_val_split(dataset_dir, class_to_index):
    val_dir = os.path.join(dataset_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Tiny-ImageNet validation images directory not found: {images_dir}")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Tiny-ImageNet validation annotations not found: {annotations_path}")

    image_paths = []
    labels = []
    with open(annotations_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            if class_name not in class_to_index:
                raise KeyError(f"Unknown Tiny-ImageNet class {class_name} in {annotations_path}")
            image_paths.append(os.path.join(images_dir, image_name))
            labels.append(class_to_index[class_name])

    return np.asarray(image_paths, dtype=object), np.asarray(labels, dtype=np.int64)


def build_tinyimagenet_client_data(args, min_samples_per_class=5):
    data_root = _resolve_tinyimagenet_data_root(args)
    dataset_dir = _ensure_tinyimagenet_extracted(data_root)
    class_to_index = _load_tinyimagenet_metadata(dataset_dir)
    train_paths, train_labels = _load_tinyimagenet_train_split(dataset_dir, class_to_index)
    val_paths, val_labels = _load_tinyimagenet_val_split(dataset_dir, class_to_index)

    client_indices, client_class_counts = split_data_dirichlet(
        labels=train_labels,
        num_clients=args.num_clients,
        alpha=args.alpha,
        min_samples_per_class=min_samples_per_class,
    )

    client_data = {}
    for client_id in range(args.num_clients):
        indices = np.asarray(client_indices[client_id], dtype=np.int64)
        np.random.shuffle(indices)
        client_data[f"client_{client_id}"] = {
            "train": {
                "data": train_paths[indices],
                "labels": train_labels[indices],
            }
        }

    shared_test = {
        "data": val_paths,
        "labels": val_labels,
    }
    client_data["shared_val_set"] = shared_test
    client_data["shared_test_set"] = shared_test

    total_allocated = sum(
        len(v["train"]["data"]) for k, v in client_data.items() if k.startswith("client_")
    )
    if total_allocated != len(train_paths):
        raise AssertionError(
            f"allocated samples {total_allocated} do not match Tiny-ImageNet training set size {len(train_paths)}"
        )

    return client_data, client_class_counts


def _iter_femnist_json_paths(all_data_dir):
    json_file_names = [
        file_name for file_name in os.listdir(all_data_dir)
        if file_name.startswith("all_data_") and file_name.endswith(".json")
    ]
    return [os.path.join(all_data_dir, file_name) for file_name in sorted(json_file_names)]


def _sample_femnist_clients(json_paths, sample_size, seed):
    rng = random.Random(seed)
    sampled_users = []
    sampled_records = {}
    total_users = 0

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as handle:
            shard = json.load(handle)

        users = shard.get("users", [])
        user_data = shard.get("user_data", {})
        for user in users:
            record = user_data.get(user)
            if record is None:
                continue

            if total_users < sample_size:
                sampled_users.append(user)
                sampled_records[user] = record
            else:
                replace_index = rng.randint(0, total_users)
                if replace_index < sample_size:
                    old_user = sampled_users[replace_index]
                    sampled_users[replace_index] = user
                    sampled_records.pop(old_user, None)
                    sampled_records[user] = record
            total_users += 1

    if total_users < sample_size:
        raise ValueError(
            f"Requested {sample_size} FEMNIST clients, but only found {total_users}."
        )

    return sampled_users, sampled_records, total_users


def _reshape_femnist_images(flattened_images):
    images = np.asarray(flattened_images, dtype=np.float32).reshape(-1, 28, 28)
    return np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8)


def _split_femnist_client_examples(images, labels, train_ratio, rng):
    sample_count = len(labels)
    if sample_count == 0:
        raise ValueError("Encountered an empty FEMNIST client after sampling.")

    if sample_count == 1:
        return images, labels, images[:0].copy(), labels[:0].copy()

    indices = list(range(sample_count))
    rng.shuffle(indices)
    train_count = int(round(sample_count * train_ratio))
    train_count = min(max(train_count, 1), sample_count - 1)

    train_indices = np.asarray(indices[:train_count], dtype=np.int64)
    test_indices = np.asarray(indices[train_count:], dtype=np.int64)
    return images[train_indices], labels[train_indices], images[test_indices], labels[test_indices]


def build_femnist_client_data(args, train_ratio=0.8):
    data_root = _resolve_femnist_data_root(args)
    all_data_dir = _ensure_femnist_extracted(data_root)
    json_paths = _iter_femnist_json_paths(all_data_dir)
    sampled_users, sampled_records, total_users = _sample_femnist_clients(
        json_paths,
        sample_size=args.num_clients,
        seed=args.seed,
    )

    if args.num_clients > total_users:
        raise ValueError(f"num_clients={args.num_clients} exceeds available FEMNIST clients={total_users}.")

    rng = random.Random(args.seed)
    client_data = {}
    client_class_counts = {}
    shared_test_images = []
    shared_test_labels = []

    shuffled_users = list(sampled_users)
    rng.shuffle(shuffled_users)
    if len(shuffled_users) <= 1:
        train_client_names = shuffled_users
        test_client_names = shuffled_users
    else:
        test_client_count = int(round(len(shuffled_users) * (1.0 - train_ratio)))
        test_client_count = min(max(test_client_count, 1), len(shuffled_users) - 1)
        test_client_names = shuffled_users[:test_client_count]
        train_client_names = shuffled_users[test_client_count:]

    for client_name in train_client_names:
        record = sampled_records[client_name]
        images = _reshape_femnist_images(record["x"])
        labels = np.asarray(record["y"], dtype=np.int64)

        if len(images) != len(labels):
            raise ValueError(f"FEMNIST client {client_name} has mismatched image/label counts.")

        client_data[client_name] = {
            "train": {"data": images, "labels": labels},
        }
        client_class_counts[client_name] = np.bincount(
            labels,
            minlength=_FEMNIST_CLASS_COUNT,
        ).astype(int)

    for client_name in test_client_names:
        record = sampled_records[client_name]
        images = _reshape_femnist_images(record["x"])
        labels = np.asarray(record["y"], dtype=np.int64)

        if len(images) != len(labels):
            raise ValueError(f"FEMNIST client {client_name} has mismatched image/label counts.")

        shared_test_images.append(images)
        shared_test_labels.append(labels)

    shared_images = np.concatenate(shared_test_images, axis=0)
    shared_labels = np.concatenate(shared_test_labels, axis=0)
    shared_split = {"data": shared_images, "labels": shared_labels}
    client_data["shared_val_set"] = shared_split
    client_data["shared_test_set"] = shared_split
    client_data["femnist_test_clients"] = list(test_client_names)
    client_data["femnist_train_clients"] = list(train_client_names)

    return client_data, client_class_counts


def build_femnist_v2_client_data(args, train_ratio=0.8):
    data_root = _resolve_femnist_data_root(args)
    all_data_dir = _ensure_femnist_extracted(data_root)
    json_paths = _iter_femnist_json_paths(all_data_dir)
    sampled_users, sampled_records, total_users = _sample_femnist_clients(
        json_paths,
        sample_size=args.num_clients,
        seed=args.seed,
    )

    if args.num_clients > total_users:
        raise ValueError(f"num_clients={args.num_clients} exceeds available FEMNIST clients={total_users}.")

    rng = random.Random(args.seed)
    client_data = {}
    client_class_counts = {}
    shared_test_images = []
    shared_test_labels = []

    for client_name in sampled_users:
        record = sampled_records[client_name]
        images = _reshape_femnist_images(record["x"])
        labels = np.asarray(record["y"], dtype=np.int64)

        if len(images) != len(labels):
            raise ValueError(f"FEMNIST client {client_name} has mismatched image/label counts.")

        train_images, train_labels, test_images, test_labels = _split_femnist_client_examples(
            images,
            labels,
            train_ratio=train_ratio,
            rng=rng,
        )

        client_data[client_name] = {
            "train": {"data": train_images, "labels": train_labels},
        }
        client_class_counts[client_name] = np.bincount(
            train_labels,
            minlength=_FEMNIST_CLASS_COUNT,
        ).astype(int)
        shared_test_images.append(test_images)
        shared_test_labels.append(test_labels)

    shared_images = np.concatenate(shared_test_images, axis=0)
    shared_labels = np.concatenate(shared_test_labels, axis=0)
    shared_split = {"data": shared_images, "labels": shared_labels}
    client_data["shared_val_set"] = shared_split
    client_data["shared_test_set"] = shared_split
    client_data["femnist_v2_clients"] = list(sampled_users)

    return client_data, client_class_counts


def _build_cifar_cache_key(dataset_name, phase, split_name, data_array):
    return (
        dataset_name,
        phase,
        split_name,
        int(data_array.__array_interface__["data"][0]),
        tuple(data_array.shape),
    )


def _build_cifar_uint8_tensor(data_array):
    tensor = torch.from_numpy(np.ascontiguousarray(data_array)).permute(0, 3, 1, 2).contiguous()
    return tensor.share_memory_()


def _build_cifar_eval_tensor(data_array, mean, std):
    tensor = torch.from_numpy(np.ascontiguousarray(data_array)).permute(0, 3, 1, 2).contiguous().float()
    tensor.div_(255.0)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
    tensor.sub_(mean_tensor).div_(std_tensor)
    return tensor.share_memory_()


class DatasetFLViT(data.Dataset):
    def __init__(self, args, phase):
        super(DatasetFLViT, self).__init__()
        self.phase = phase
        self.args = args
        self.cached_tensor = None
        self.use_cifar_tensor_cache = bool(getattr(args, "enable_cifar_tensor_cache", False))
        self._cifar_mean = None
        self._cifar_std = None

        if args.dataset in ["cifar10", "cifar100"]:
            mean = (0.4914, 0.4822, 0.4465) if args.dataset == "cifar10" else (0.5071, 0.4867, 0.4408)
            std = (0.2470, 0.2435, 0.2616) if args.dataset == "cifar10" else (0.2675, 0.2565, 0.2761)
            self._cifar_mean = mean
            self._cifar_std = std
            if self.phase == "train":
                enable_data_augmentation = getattr(args, "enable_data_augmentation", True)
                if self.use_cifar_tensor_cache:
                    if enable_data_augmentation:
                        self.transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ConvertImageDtype(torch.float32),
                            transforms.Normalize(mean, std),
                        ])
                    else:
                        self.transform = transforms.Compose([
                            transforms.ConvertImageDtype(torch.float32),
                            transforms.Normalize(mean, std),
                        ])
                else:
                    if enable_data_augmentation:
                        self.transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
                    else:
                        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
            else:
                if self.use_cifar_tensor_cache:
                    self.transform = None
                else:
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])
        elif args.dataset in ["femnist", "femnist-v2"]:
            is_ondev_femnist = (
                args.dataset == "femnist"
                and getattr(args, "model", "") in ["OnDev-LCT-8/1", "OnDev-LCT-4/1"]
            )
            femnist_transforms = []
            if is_ondev_femnist:
                target_size = int(getattr(args, "img_size", 28))
                femnist_transforms.append(transforms.Resize((target_size, target_size)))
            femnist_transforms.extend([
                transforms.ToTensor(),
                transforms.Lambda(lambda tensor: tensor.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.transform = transforms.Compose(femnist_transforms)
        elif args.dataset == "tinyimagenet":
            target_size = int(getattr(args, "img_size", 64))
            tinyimagenet_transforms = []
            if self.phase == "train":
                if getattr(args, "enable_data_augmentation", True):
                    tinyimagenet_transforms.extend([
                        transforms.RandomResizedCrop(
                            target_size,
                            scale=(0.8, 1.0),
                            interpolation=transforms.InterpolationMode.BILINEAR,
                        ),
                        transforms.RandomHorizontalFlip(),
                    ])
                elif target_size != 64:
                    tinyimagenet_transforms.append(
                        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR)
                    )
            elif target_size != 64:
                tinyimagenet_transforms.append(
                    transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR)
                )

            tinyimagenet_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.transform = transforms.Compose(tinyimagenet_transforms)
        else:
            self.transform = build_transform(self.phase == "train", args)

        if args.dataset in ["cifar10", "cifar100", "femnist", "femnist-v2", "tinyimagenet", "CelebA"]:
            client_data = args.client_data
            if self.phase == "train":
                if args.dataset in ["cifar10", "cifar100", "femnist", "femnist-v2", "tinyimagenet"]:
                    self.data = client_data[args.single_client]["train"]["data"]
                    self.labels = client_data[args.single_client]["train"]["labels"]
            elif self.phase == "val":
                if args.dataset in ["cifar10", "cifar100", "femnist", "femnist-v2", "tinyimagenet"]:
                    self.data = client_data["shared_val_set"]["data"]
                    self.labels = client_data["shared_val_set"]["labels"]
            else:
                if args.dataset in ["cifar10", "cifar100", "femnist", "femnist-v2", "tinyimagenet"]:
                    test_key = "shared_test_set" if "shared_test_set" in client_data else "shared_val_set"
                    self.data = client_data[test_key]["data"]
                    self.labels = client_data[test_key]["labels"]

            if args.dataset in ["cifar10", "cifar100"] and self.use_cifar_tensor_cache:
                if self.phase == "train":
                    split_name = args.single_client
                    cache_key = _build_cifar_cache_key(args.dataset, self.phase, split_name, self.data)
                    cached_tensor = _CIFAR_UINT8_CACHE.get(cache_key)
                    if cached_tensor is None:
                        cached_tensor = _build_cifar_uint8_tensor(self.data)
                        _CIFAR_UINT8_CACHE[cache_key] = cached_tensor
                    self.cached_tensor = cached_tensor
                else:
                    split_name = "shared_val_set" if self.phase == "val" else test_key
                    cache_key = _build_cifar_cache_key(args.dataset, self.phase, split_name, self.data)
                    cached_tensor = _CIFAR_EVAL_CACHE.get(cache_key)
                    if cached_tensor is None:
                        cached_tensor = _build_cifar_eval_tensor(self.data, self._cifar_mean, self._cifar_std)
                        _CIFAR_EVAL_CACHE[cache_key] = cached_tensor
                    self.cached_tensor = cached_tensor

        elif args.dataset == "Retina":
            if self.phase == "test":
                args.single_client = os.path.join(args.data_path, "test.csv")
            elif self.phase == "val":
                args.single_client = os.path.join(args.data_path, "val.csv")

            cur_client_path = os.path.join(args.data_path, args.split_type, args.single_client)
            self.img_paths = list({line.strip().split(",")[0] for line in open(cur_client_path)})
            self.labels = {
                line.strip().split(",")[0]: float(line.strip().split(",")[1])
                for line in open(os.path.join(args.data_path, "labels.csv"))
            }
            args.loadSize = 256
            args.fineSize_w = 224
            args.fineSize_h = 224
            self.transform = None

    def __getitem__(self, index):
        if self.args.dataset in ["cifar10", "cifar100"]:
            target = int(self.labels[index])
            if self.cached_tensor is not None:
                img = self.cached_tensor[index]
            else:
                img = self.data[index]
                img = Image.fromarray(img)

        elif self.args.dataset in ["femnist", "femnist-v2"]:
            target = int(self.labels[index])
            img = Image.fromarray(self.data[index], mode="L")

        elif self.args.dataset == "tinyimagenet":
            target = int(self.labels[index])
            img = Image.open(self.data[index]).convert("RGB")

        elif self.args.dataset == "CelebA":
            name = self.data[index]
            target = self.labels[name]
            path = os.path.join(self.args.data_path, "img_align_celeba", name)
            img = Image.open(path).convert("RGB")
            target = np.asarray(target).astype("int64")

        elif self.args.dataset == "Retina":
            index = index % len(self.img_paths)
            path = os.path.join(self.args.data_path, "train-all", self.img_paths[index])
            name = self.img_paths[index]
            img = np.load(path)
            target = np.asarray(self.labels[name]).astype("int64")

            if self.phase == "train":
                if random.random() < 0.5:
                    img = np.fliplr(img).copy()
                else:
                    img = np.array(img)
                img = resize(img, (self.args.loadSize, self.args.loadSize))
                w_offset = random.randint(0, max(0, self.args.loadSize - self.args.fineSize_w - 1))
                h_offset = random.randint(0, max(0, self.args.loadSize - self.args.fineSize_h - 1))
                img = img[
                    w_offset : w_offset + self.args.fineSize_w,
                    h_offset : h_offset + self.args.fineSize_h,
                ]
            else:
                img = resize(img, (self.args.loadSize, self.args.loadSize))
                img = np.array(img)
                img = img[
                    (self.args.loadSize - self.args.fineSize_w) // 2 : (self.args.loadSize - self.args.fineSize_w) // 2 + self.args.fineSize_w,
                    (self.args.loadSize - self.args.fineSize_h) // 2 : (self.args.loadSize - self.args.fineSize_h) // 2 + self.args.fineSize_h,
                ]

            img = torch.from_numpy(img).float()
            img = (1 + 1) / 255 * (img - 255) + 1
            if img.dim() < 3:
                img = torch.stack((img, img, img))
            else:
                img = img.permute(2, 1, 0)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.args.dataset == "Retina":
            return len(self.img_paths)
        return len(self.data)


def create_dataset_and_evalmetrix(args):
    if args.dataset in ["cifar10", "cifar100"]:
        args.client_data, args.client_class_counts = build_client_data(
            dataset_name=args.dataset,
            data_path="./data",
            num_clients=args.num_clients,
            alpha=args.alpha,
            min_samples_per_class=5,
        )
        client_data = args.client_data
        args.dis_cvs_files = [
            name for name in client_data.keys() if name not in ["shared_val_set", "shared_test_set"]
        ]

        args.clients_with_len = {}
        for client_name, client_item in client_data.items():
            if client_name not in ["shared_val_set", "shared_test_set"] and "train" in client_item:
                args.clients_with_len[client_name] = len(client_item["train"]["data"])

    elif args.dataset == "femnist":
        args.client_data, args.client_class_counts = build_femnist_client_data(args)
        client_data = args.client_data
        args.dis_cvs_files = [
            name for name in client_data.keys() if name not in ["shared_val_set", "shared_test_set", "femnist_test_clients", "femnist_train_clients"]
        ]
        args.clients_with_len = {
            client_name: len(client_item["train"]["data"])
            for client_name, client_item in client_data.items()
            if client_name not in ["shared_val_set", "shared_test_set", "femnist_test_clients", "femnist_train_clients"] and "train" in client_item
        }

    elif args.dataset == "femnist-v2":
        args.client_data, args.client_class_counts = build_femnist_v2_client_data(args)
        client_data = args.client_data
        args.dis_cvs_files = [
            name for name in client_data.keys() if name not in ["shared_val_set", "shared_test_set", "femnist_v2_clients"]
        ]
        args.clients_with_len = {
            client_name: len(client_item["train"]["data"])
            for client_name, client_item in client_data.items()
            if client_name not in ["shared_val_set", "shared_test_set", "femnist_v2_clients"] and "train" in client_item
        }

    elif args.dataset == "tinyimagenet":
        args.client_data, args.client_class_counts = build_tinyimagenet_client_data(args)
        client_data = args.client_data
        args.dis_cvs_files = [
            name for name in client_data.keys() if name not in ["shared_val_set", "shared_test_set"]
        ]
        args.clients_with_len = {
            client_name: len(client_item["train"]["data"])
            for client_name, client_item in client_data.items()
            if client_name not in ["shared_val_set", "shared_test_set"] and "train" in client_item
        }

    elif args.dataset == "Retina":
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
        args.clients_with_len = {}
        for single_client in args.dis_cvs_files:
            img_paths = list({
                line.strip().split(",")[0]
                for line in open(os.path.join(args.data_path, args.split_type, single_client))
            })
            args.clients_with_len[single_client] = len(img_paths)

    elif args.dataset == "CelebA":
        data_all = np.load(os.path.join("./data/", args.dataset + ".npy"), allow_pickle=True)
        data_all = data_all.items()
        args.dis_cvs_files = list(data_all[args.split_type]["train"].keys())
        if getattr(args, "split_type", None) == "real":
            args.clients_with_len = {
                name: len(data_all["real"]["train"][name]["x"]) for name in data_all["real"]["train"]
            }

    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = 0.0
        args.current_test_acc[single_client] = 0.0
        args.best_eval_loss[single_client] = 9999
