import os
import tempfile
import torch
import pytest
from collections import Counter

from app.utils import (
    update_correct_per_class,
    update_correct_per_class_topk,
    update_correct_per_class_avgk,
    count_correct_topk,
    count_correct_avgk,
    save,
    load_model,
    load_optimizer,
    decay_lr,
    update_optimizer,
    get_model,
)

# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def small_batch():
    # 3 classes, batch of 4
    scores = torch.tensor([
        [0.2, 0.5, 0.3],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.3, 0.4, 0.3],
    ])
    labels = torch.tensor([1, 0, 1, 2])
    return scores, labels

@pytest.fixture
def tmp_checkpoint(tmp_path):
    # Create a dummy model and optimizer, perform a step, then save checkpoint
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.lin(x)

    model = DummyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(1, 3)
    loss = model(x).sum()
    loss.backward()
    optim.step()
    ckpt = tmp_path / "ckpt.pth"
    save(model, optim, epoch=5, location=str(ckpt))
    return ckpt

@pytest.fixture
def optimizer():
    params = [torch.nn.Parameter(torch.randn(2,2))]
    return torch.optim.SGD(params, lr=0.5)

# ---------------------
# Metric Tests
# ---------------------

def test_update_correct_per_class(small_batch):
    scores, labels = small_batch
    d = {0: 0, 1: 0, 2: 0}
    update_correct_per_class(scores, labels, d)
    assert d == {0: 1, 1: 2, 2: 0}


# def test_update_correct_per_class_topk(small_batch):
#     scores, labels = small_batch
#     d = {0: 0, 1: 0, 2: 0}
#     update_correct_per_class_topk(scores, labels, d, k=2)
#     assert d[0] == 1
#     assert d[1] == 2
#     assert d[2] == 1


# def test_count_correct_topk(small_batch):
#     scores, labels = small_batch
#     assert count_correct_topk(scores, labels, k=2) == 4


def test_count_correct_topk_k1(small_batch):
    scores, labels = small_batch
    assert count_correct_topk(scores, labels, k=1) == 3


def test_count_correct_topk_full(small_batch):
    scores, labels = small_batch
    assert count_correct_topk(scores, labels, k=3) == 4


def test_update_correct_per_class_avgk(small_batch):
    scores, labels = small_batch
    d = {0: 0, 1: 0, 2: 0}
    update_correct_per_class_avgk(scores, labels, d, lmbda=0.4)
    assert d == {0: 1, 1: 2, 2: 0}


def test_count_correct_avgk(small_batch):
    scores, labels = small_batch
    assert count_correct_avgk(scores, labels, lmbda=0.4) == 3

# ---------------------
# Checkpoint Tests
# ---------------------

def test_load_model_and_optimizer(tmp_checkpoint):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.lin(x)

    model = DummyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    epoch = load_model(model, str(tmp_checkpoint), use_gpu=False)
    assert epoch == 5
    load_optimizer(optim, str(tmp_checkpoint), use_gpu=False)


def test_load_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(torch.nn.Linear(1,1), str(tmp_path / "nope.pth"), use_gpu=False)

# ---------------------
# Scheduler Tests
# ---------------------

def test_decay_lr(optimizer):
    old_lr = optimizer.param_groups[0]["lr"]
    decay_lr(optimizer)
    assert pytest.approx(optimizer.param_groups[0]["lr"]) == old_lr * 0.1


def test_update_optimizer(optimizer):
    schedule = {3, 5, 7}
    opt1 = update_optimizer(optimizer, schedule, epoch=1)
    assert opt1.param_groups[0]["lr"] == 0.5
    opt3 = update_optimizer(optimizer, schedule, epoch=3)
    assert pytest.approx(opt3.param_groups[0]["lr"]) == 0.05

# ---------------------
# load_optimizer Only Test
# ---------------------

def test_load_optimizer_only(tmp_checkpoint):
    ckpt = str(tmp_checkpoint)
    data = torch.load(ckpt, map_location="cpu")
    orig_opt_sd = data["optimizer"]
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.lin(x)

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    load_optimizer(opt, ckpt, use_gpu=False)
    new_sd = opt.state_dict()
    assert new_sd['param_groups'] == orig_opt_sd['param_groups']

# ---------------------
# get_model Tests
# ---------------------

def make_args(model_name, pretrained=False):
    class Args: pass
    args = Args()
    args.model = model_name
    args.pretrained = pretrained
    return args

@pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50"])
def test_get_model_pytorch_non_pretrained(model_name):
    args = make_args(model_name, pretrained=False)
    model = get_model(args, n_classes=10)
    # For torchvision models, verify output layer matches n_classes
    if hasattr(model, 'fc'):
        assert model.fc.out_features == 10
    elif hasattr(model, 'classifier'):
        # e.g., squeezenet or vgg
        assert any(layer.out_features == 10 for layer in model.classifier if hasattr(layer, 'out_features'))

# @pytest.mark.parametrize("model_name", ["resnet18", "alexnet", "densenet121"])
# def test_get_model_pytorch_pretrained(model_name):
#     args = make_args(model_name, pretrained=True)
#     model = get_model(args, n_classes=8)
#     # Check classifier adapted to n_classes
#     if model_name.startswith('resnet') or model_name.startswith('wide_resnet'):
#         assert model.fc.out_features == 8
#     elif model_name.startswith('alexnet') or model_name.startswith('vgg11'):
#         assert model.classifier[6].out_features == 8
#     elif model_name.startswith('densenet'):
#         assert model.classifier.out_features == 8


def test_get_model_inception_v3():
    args = make_args("inception_v3", pretrained=False)
    model = get_model(args, n_classes=5)
    # Inception v3 uses fc
    assert model.fc.out_features == 5


# def test_get_model_timm():
#     args = make_args("vit_base_patch16_224", pretrained=True)
#     model = get_model(args, n_classes=12)
#     # timm sets num_classes attribute
#     assert getattr(model, 'num_classes', None) == 12
