"""Image datasets and DINO-style multi-crop transforms."""

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import InterpolationMode, v2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _normalize():
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]


def _augment(resolution, scale, recipe):
    minimal = recipe == "minimal"
    jitter = (0.8, 0.8, 0.8, 0.2) if minimal else (0.4, 0.4, 0.2, 0.1)
    return v2.Compose([
        v2.RandomResizedCrop(
            resolution, scale=scale,
            interpolation=InterpolationMode.BILINEAR if minimal else InterpolationMode.BICUBIC,
        ),
        v2.ToImage(),
        v2.RandomApply([v2.ColorJitter(*jitter)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.5),
        v2.RandomSolarize(threshold=128, p=0.1 if minimal else 0.2),
        v2.RandomHorizontalFlip(),
        *_normalize()[1:],
    ])


def evaluation_transform(resolution, recipe):
    minimal = recipe == "minimal"
    resize = resolution if minimal else round(resolution / 0.875)
    interpolation = InterpolationMode.BILINEAR if minimal else InterpolationMode.BICUBIC
    return v2.Compose([v2.Resize(resize, interpolation=interpolation), v2.CenterCrop(resolution), *_normalize()])


def _folder(root, train):
    split = root / ("train" if train else "val")
    return datasets.ImageFolder(split)


def image_dataset(name, root, train, download=False):
    """Load official torchvision splits or user-supplied ImageFolder splits."""
    root = Path(root).expanduser()
    if name == "fake":
        return datasets.FakeData(size=16 if train else 8, image_size=(3, 48, 48),
                                 num_classes=4, random_offset=0 if train else 16)
    if (root / "train").is_dir() or name in {"inet1k", "inet100", "galaxy10", "imagefolder"}:
        return _folder(root, train)
    if name == "imagenette":
        return datasets.Imagenette(root, split="train" if train else "val", size="160px", download=download)
    if name in {"cifar10", "cifar100"}:
        factory = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        return factory(root, train=train, download=download)
    if name in {"flowers102", "food101", "dtd", "aircraft", "cars", "pets"}:
        factory, train_split, test_split = {
            "flowers102": (datasets.Flowers102, "train", "test"),
            "food101": (datasets.Food101, "train", "test"),
            "dtd": (datasets.DTD, "train", "test"),
            "aircraft": (datasets.FGVCAircraft, "trainval", "test"),
            "cars": (datasets.StanfordCars, "train", "test"),
            "pets": (datasets.OxfordIIITPet, "trainval", "test"),
        }[name]
        return factory(root, split=train_split if train else test_split, download=download)
    raise ValueError(f"Unknown image dataset: {name}")


def check_splits(train, test):
    if hasattr(train, "class_to_idx") and train.class_to_idx != getattr(test, "class_to_idx", None):
        raise ValueError("Train and validation splits must use the same class-to-index mapping")
    if not len(train) or not len(test):
        raise ValueError("Both dataset splits must contain images")


def class_count(dataset):
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if hasattr(dataset, "num_classes"):
        return dataset.num_classes
    return max(targets(dataset)) + 1


def targets(dataset):
    for key in ("targets", "_labels"):
        values = getattr(dataset, key, None)
        if values is not None:
            return values
    samples = getattr(dataset, "_samples", None)
    if samples is not None:  # torchvision.datasets.Imagenette
        return [label for _, label in samples]
    return [dataset[index][1] for index in range(len(dataset))]


def few_shot_subset(dataset, shots, seed):
    if shots is None:
        return dataset
    if shots < 1:
        raise ValueError("shots must be positive")
    labels = torch.as_tensor(targets(dataset))
    rng = torch.Generator().manual_seed(seed)
    indices = []
    for label in labels.unique():
        members = torch.where(labels == label)[0]
        selected = members[torch.randperm(len(members), generator=rng)[:shots]]
        indices.extend(selected.tolist())
    return Subset(dataset, indices)


class TransformedImages(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image.convert("RGB")), label


def view_transform(config):
    if config.augment == "minimal":
        global_scale = (0.08, 1.0)
    else:
        global_scale = (0.3, 1.0) if config.local_views else (0.08, 1.0)
    groups = [(_augment(config.resolution, global_scale, config.augment), config.global_views)]
    if config.local_views:
        groups.append((_augment(config.local_resolution, (0.05, 0.3), config.augment), config.local_views))
    return MultiCrop(groups)


class MultiCrop:
    def __init__(self, groups):
        self.groups = groups

    def __call__(self, image):
        return tuple(transform(image) for transform, count in self.groups for _ in range(count))
