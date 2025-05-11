import pytest
import torch.nn as nn
import timm

from app.utils  import get_model  # adjust this import to match your module path

class Args:
    def __init__(self, model, pretrained=False):
        self.model = model
        self.pretrained = pretrained


def test_non_pretrained_general():
    args = Args(model="resnet18", pretrained=False)
    n_classes = 10
    model = get_model(args, n_classes)
    # For resnet variants, check the final fully-connected layer
    assert hasattr(model, "fc"), "Expected .fc attribute on non-pretrained ResNet"
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == n_classes


def test_non_pretrained_inception():
    args = Args(model="inception_v3", pretrained=False)
    n_classes = 7
    model = get_model(args, n_classes)
    # Inception-specific settings: no AuxLogits, correct number of classes
    assert model.AuxLogits is None
    assert model.fc.out_features == n_classes


@pytest.mark.parametrize("model_name, check_fn", [
    ("resnet50", lambda m, n: m.fc.out_features == n),
    ("alexnet", lambda m, n: m.classifier[6].out_features == n),
    ("densenet121", lambda m, n: m.classifier.out_features == n),
    ("mobilenet_v2", lambda m, n: m.classifier[1].out_features == n),
    ("inception_v3", lambda m, n: (m.fc.out_features == n and m.AuxLogits is None)),
    ("squeezenet", lambda m, n: (m.num_classes == n and m.classifier[1].out_channels == n)),
    ("mobilenet_v3_large", lambda m, n: m.classifier[-1].out_features == n),
])
def test_pretrained_torch_models(model_name, check_fn):
    args = Args(model=model_name, pretrained=True)
    n_classes = 12
    model = get_model(args, n_classes)
    assert check_fn(model, n_classes), f"Branch for {model_name} did not set up classifier correctly"


def test_timm_model(monkeypatch):
    called = {}
    def fake_create_model(name, pretrained, num_classes):
        called['args'] = (name, pretrained, num_classes)
        return nn.Identity()

    monkeypatch.setattr(timm, "create_model", fake_create_model)
    args = Args(model="efficientnet_b0", pretrained=True)
    n_classes = 5
    model = get_model(args, n_classes)
    assert isinstance(model, nn.Identity)
    assert called['args'] == ("efficientnet_b0", True, n_classes)


def test_unknown_model_raises():
    args = Args(model="not_a_model", pretrained=False)
    with pytest.raises(NotImplementedError):
        get_model(args, 3)
