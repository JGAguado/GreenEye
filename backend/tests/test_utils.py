import os
from collections import Counter

import pytest
import torch
import torch.nn as nn

from app.utils import (
    get_data,
    get_model,
    load_model,
    save,
    set_seed,
    update_correct_per_class,
)


class Args:
    def __init__(self, model, pretrained, seed=None):
        self.model = model
        self.pretrained = pretrained
        self.seed = seed


@pytest.fixture
def dummy_model():
    return nn.Linear(10, 2)


@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.SGD(dummy_model.parameters(), lr=0.01)


@pytest.fixture
def temp_file(tmp_path):
    return os.path.join(tmp_path, "temp.pth")


def test_get_model():
    args = Args(model="resnet18", pretrained=False)
    model = get_model(args, n_classes=10)
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 10


def test_set_seed():
    args = Args(model=None, pretrained=None, seed=42)
    set_seed(args, use_gpu=False)

    # Generate a sequence of random numbers and verify consistency
    random_numbers = [torch.randint(0, 100, (1,)).item() for _ in range(5)]

    # Reset the seed and generate the same sequence
    set_seed(args, use_gpu=False)
    repeated_random_numbers = [torch.randint(0, 100, (1,)).item() for _ in range(5)]

    # Assert that the sequences are identical
    assert random_numbers == repeated_random_numbers


def test_load_model(dummy_model, temp_file):
    torch.save({"model": dummy_model.state_dict(), "epoch": 5}, temp_file)
    epoch = load_model(dummy_model, temp_file, use_gpu=False)
    assert epoch == 5


def test_save(dummy_model, dummy_optimizer, temp_file):
    save(dummy_model, dummy_optimizer, epoch=3, location=temp_file)
    assert os.path.exists(temp_file)
    checkpoint = torch.load(temp_file)
    assert checkpoint["epoch"] == 3
    assert "model" in checkpoint
    assert "optimizer" in checkpoint


def test_update_correct_per_class():
    batch_output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    batch_y = torch.tensor([1, 0])
    d = Counter()
    update_correct_per_class(batch_output, batch_y, d)
    assert d[0] == 1
    assert d[1] == 1


def test_get_data(tmp_path):
    # Create dummy dataset structure
    root = tmp_path / "dataset"
    os.makedirs(root / "train" / "class1")
    os.makedirs(root / "val" / "class1")
    os.makedirs(root / "test" / "class1")
    with open(root / "train" / "class1" / "img1.jpg", "w") as f:
        f.write("dummy")
    with open(root / "val" / "class1" / "img1.jpg", "w") as f:
        f.write("dummy")
    with open(root / "test" / "class1" / "img1.jpg", "w") as f:
        f.write("dummy")

    trainloader, valloader, testloader, dataset_attributes = get_data(
        root=str(root),
        image_size=224,
        crop_size=224,
        batch_size=1,
        num_workers=0,
        pretrained=False,
    )

    assert len(trainloader) == 1
    assert len(valloader) == 1
    assert len(testloader) == 1
    assert dataset_attributes["n_classes"] == 1
