import pytest
import torch
from torch import nn

from worldlab.nn import EppsPulley, LeJEPA, LeJEPAObjective, SIGReg


def test_epps_pulley_matches_complex_trapezoid():
    samples = torch.randn(3, 13, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    grid = torch.linspace(-3, 3, 33, dtype=samples.dtype)
    window = (-grid.square() / 2).exp()
    cf = (1j * samples.unsqueeze(-1) * grid).exp().mean(dim=1)
    expected = 13 * torch.trapezoid((cf - window).abs().square() * window, grid)
    torch.testing.assert_close(EppsPulley(17, 3)(samples), expected)
    torch.testing.assert_close(EppsPulley(33, 3, symmetric_grid=True)(samples), expected)


def test_sigreg_uses_independent_view_batches_and_preserves_gradients_with_chunking():
    generator = torch.Generator().manual_seed(9)
    views = torch.randn(3, 16, 5, dtype=torch.float64, generator=generator, requires_grad=True)
    regularizer = SIGReg(13, chunk_size=4, seed=5)
    directions = regularizer.directions(views, step=2)
    torch.testing.assert_close(directions.norm(dim=0), torch.ones(13, dtype=views.dtype))
    assert not torch.equal(directions, regularizer.directions(views, step=3))
    actual = regularizer(views, step=2)
    expected = torch.stack([regularizer.test(view @ directions).mean() for view in views]).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, views)[0], torch.autograd.grad(expected, views)[0])


def test_sigreg_prefers_standard_normal_to_constant_or_shifted_samples():
    normal = torch.randn(512, 6, generator=torch.Generator().manual_seed(9))
    sigreg = SIGReg(24)
    reference = sigreg(normal, step=1)
    assert sigreg(torch.zeros_like(normal), step=1) > reference
    assert sigreg(normal + 2, step=1) > reference


def test_global_center_gradient_flows_to_target():
    projections = torch.tensor([[[1.0]], [[3.0]], [[7.0]]], requires_grad=True)
    objective = LeJEPAObjective(0.0, SIGReg(2))
    result = objective(projections, num_global=2, step=0)
    torch.testing.assert_close(result.loss, torch.tensor(9.0))
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, projections)[0].flatten(),
        torch.tensor([-7 / 3, -1.0, 10 / 3]),
    )


def test_lejepa_groups_crop_shapes_and_preserves_view_order():
    model = LeJEPA(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    views = [torch.full((2, 1, size, size), float(index)) for index, size in enumerate((8, 8, 4))]
    features, projections = model.encode_views(views)
    assert features.shape == projections.shape == (3, 2, 1)
    torch.testing.assert_close(projections[:, 0, 0], torch.arange(3, dtype=torch.float32))


def test_multicrop_classifier_and_label_isolation():
    pytest.importorskip("timm")
    from worldlab.experiments.lejepa.config import load_config
    from worldlab.experiments.lejepa.vision import LeJEPAClassifier

    cfg = load_config("smoke", overrides=(
        "backbone=vit_tiny_patch8_224", "local_views=2", "local_resolution=16",
        "predictor=true", "swa=true", "patch_mask_ratio=0.3", "register_tokens=2",
        "activation_checkpointing=true",
    ))
    model = LeJEPAClassifier(cfg, 4)
    model.update_teacher()
    views = [torch.randn(2, 3, size, size) for size in (32, 32, 16, 16)]
    output, logits = model(views, 0)
    assert logits.shape == (2, 2, 4)
    logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.jepa.parameters())
    output.loss.backward()
    assert model.jepa.encoder.backbone.patch_embed.proj.weight.grad is not None
    assert model.predictor[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    pytest.importorskip("timm")
    from worldlab.experiments.lejepa import train as training
    from worldlab.experiments.lejepa.config import load_config

    cfg = load_config("smoke", overrides=("epochs=2",))
    baseline = training.train(cfg, tmp_path / "baseline", "cpu")
    original = training._write_history

    def stop_after_first_checkpoint(output, history):
        original(output, history)
        raise InterruptedError

    with monkeypatch.context() as patch:
        patch.setattr(training, "_write_history", stop_after_first_checkpoint)
        with pytest.raises(InterruptedError):
            training.train(cfg, tmp_path / "resumed", "cpu")
    resumed = training.train(cfg, tmp_path / "resumed", "cpu", resume=True)
    assert resumed == baseline
    left = torch.load(tmp_path / "baseline" / "last.pt", weights_only=True)["model"]
    right = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)["model"]
    torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_probe_uses_head_feature_dim_when_it_differs_from_backbone_channels():
    pytest.importorskip("timm")
    from worldlab.experiments.lejepa.config import load_config
    from worldlab.experiments.lejepa.vision import LeJEPAClassifier

    cfg = load_config("smoke", overrides=("backbone=mobilenetv3_small_100",))
    model = LeJEPAClassifier(cfg, 4).eval()
    images = torch.randn(2, 3, cfg.resolution, cfg.resolution)
    with torch.inference_mode():
        loss, logits = model((images, images), 0)
        assert loss.loss.isfinite()
        assert logits.shape == (2, 2, 4)
