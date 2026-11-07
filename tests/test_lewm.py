import pytest
import torch
from torch import nn

from worldlab.nn import ActionConditionedPredictor, LeWorldModel, SIGReg
from worldlab.planning import cem, gradient_plan


class AddAction(nn.Module):
    num_frames = 3

    def forward(self, embeddings, actions):
        return embeddings + actions


def test_rollout_aligns_executed_and_future_actions_and_keeps_gradients():
    model = LeWorldModel(nn.Identity(), AddAction(), nn.Identity()).eval()
    context = torch.tensor([[[0.0], [2.0]]])
    future = torch.tensor([[[[3.0], [5.0]], [[4.0], [6.0]]]], requires_grad=True)
    past = torch.tensor([[[2.0]]])
    predictions = model.rollout(context, future, past)
    torch.testing.assert_close(predictions[..., 0], torch.tensor([[[0., 2., 5., 10.], [0., 2., 6., 12.]]]))
    cost = model.goal_cost(context, torch.zeros(1, 1), future, past)
    torch.testing.assert_close(cost, torch.tensor([[100., 144.]]))
    torch.testing.assert_close(torch.autograd.grad(cost.sum(), future)[0][0, :, :, 0],
                               torch.tensor([[20., 20.], [24., 24.]]))
    with pytest.raises(ValueError, match="executed action blocks"):
        model.rollout(context, future)


def test_predictor_is_causal_with_action_conditioning():
    torch.manual_seed(0)
    predictor = ActionConditionedPredictor(num_frames=3, input_dim=4, hidden_dim=4, depth=2,
                                           heads=2, dim_head=2, mlp_dim=8).eval()
    for block in predictor.transformer.layers:
        nn.init.normal_(block.adaLN_modulation[-1].weight)
    latents, actions = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    expected = predictor(latents, actions)
    latents[:, 2] += 1
    actions[:, 2] += 2
    actual = predictor(latents, actions)
    torch.testing.assert_close(actual[:, :2], expected[:, :2])
    assert not torch.equal(actual[:, 2], expected[:, 2])


def test_loss_uses_stepwise_sigreg_and_a_differentiable_next_frame():
    model = LeWorldModel(nn.Identity(), AddAction(), nn.Identity(), sigreg=SIGReg(4), weight=0.09)
    frames = torch.randn(2, 4, 3, requires_grad=True)
    actions = torch.zeros(2, 4, 3)
    actions[:, -1] = float("nan")  # The last frame has no prediction target after it.
    result = model(frames, actions, step=1)
    expected = (frames[:, :-1] - frames[:, 1:]).square().mean()
    torch.testing.assert_close(result.prediction, expected)
    torch.testing.assert_close(result.sigreg, model.sigreg(frames.transpose(0, 1), step=1))
    torch.testing.assert_close(result.loss, result.prediction + 0.09 * result.sigreg)
    assert torch.autograd.grad(result.prediction, frames)[0][:, -1].abs().sum() > 0


def test_planners_reduce_a_batched_cost():
    target = torch.tensor([0.3, -0.7]).view(2, 1, 1, 1)

    def cost(candidates):
        return (candidates - target).square().mean(dim=(-1, -2))

    initial = torch.zeros(2, 2, 1)
    plan = cem(cost, initial, samples=40, elites=6, iterations=10,
               generator=torch.Generator().manual_seed(6))
    gradient = gradient_plan(cost, initial, samples=10, iterations=20,
                             generator=torch.Generator().manual_seed(7))
    assert (cost(plan[:, None]) < cost(initial[:, None]) / 20).all()
    assert (cost(gradient[:, None]) < cost(initial[:, None]) / 20).all()


def test_hdf5_windows_do_not_cross_episodes_and_group_raw_actions(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    pytest.importorskip("torchvision")
    from worldlab.experiments.lewm.data import Trajectories, split_dataset

    path = tmp_path / "clips.h5"
    with h5py.File(path, "w") as f:
        f["pixels"] = torch.zeros(16, 8, 8, 3, dtype=torch.uint8).numpy()
        f["action"] = torch.arange(16).reshape(-1, 1).numpy()
        f["ep_offset"], f["ep_len"] = [0, 8], [8, 8]
    dataset = Trajectories(path, 8, history=2, frame_skip=2)
    assert len(dataset) == 6
    first_window_second_episode = dataset[3]
    assert first_window_second_episode["pixels"].shape == (3, 3, 8, 8)
    recovered = first_window_second_episode["action"] * dataset.std + dataset.mean
    torch.testing.assert_close(recovered.flatten(), torch.arange(8, 14).float())
    train, val = split_dataset(dataset, 0.5, 3, "episodes")
    assert set(index // 3 for index in train.indices).isdisjoint(index // 3 for index in val.indices)


def test_resume_at_epoch_boundary_matches_continuous_training(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    pytest.importorskip("torchvision")
    pytest.importorskip("yaml")
    from worldlab.experiments.lewm import train as training
    from worldlab.experiments.lewm.config import load_config

    generator = torch.Generator().manual_seed(0)
    path = tmp_path / "clips.h5"
    with h5py.File(path, "w") as f:
        f["pixels"] = torch.randint(0, 256, (24, 8, 8, 3), dtype=torch.uint8, generator=generator).numpy()
        f["action"] = torch.randn(24, 2, generator=generator).numpy()
        f["ep_offset"], f["ep_len"] = [0, 8, 16], [8, 8, 8]
    config = load_config("smoke", overrides=("epochs=2",))
    baseline = training.train(config, path, tmp_path / "baseline", "cpu")
    original = training._write_metrics

    def interrupt(output, history):
        original(output, history)
        raise InterruptedError

    with monkeypatch.context() as patch:
        patch.setattr(training, "_write_metrics", interrupt)
        with pytest.raises(InterruptedError):
            training.train(config, path, tmp_path / "resume", "cpu")
    resumed = training.train(config, path, tmp_path / "resume", "cpu", resume=True)
    assert resumed == baseline
    before = torch.load(tmp_path / "baseline" / "last.pt", weights_only=True)["model"]
    after = torch.load(tmp_path / "resume" / "last.pt", weights_only=True)["model"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
