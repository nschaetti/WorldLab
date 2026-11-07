from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldlab.nn import PlatonicJEPA
from worldlab.representation import Rotation, mutual_knn, paired_alignment, prh_score, scene_explained_fraction
from worldlab.representation.spectral import mismatch_bound, spectral_solution


def test_prh_pools_views_once_for_projector_and_sigreg():
    model = PlatonicJEPA(nn.Flatten(), 2, projection_dim=3, projection_width=5, num_slices=4)
    first = torch.randn(4, 2, 1)
    second = first.transpose(1, 2)  # Different shape forces a separate encoder call.
    batches = []
    handle = model.projector.register_forward_pre_hook(lambda _, args: batches.append(len(args[0])))
    result = model((first, second), 1, step=2)
    handle.remove()
    assert batches == [8]
    _, projections = model.encode_views((first, second))
    expected = model.objective.sigreg(projections.flatten(0, 1), step=2)
    torch.testing.assert_close(result.sigreg, expected)
    torch.testing.assert_close(result.loss, result.prediction + 0.02 * expected)


def test_caption_views_share_a_scene_or_a_single_caption():
    from worldlab.experiments.prh.views import TextViews

    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2, mask_token_id=3, pad_token_id=0)
    views = TextViews(tokenizer, length=8, mask=0.0)
    captions = [torch.tensor([11, 11]), torch.tensor([19, 19])]
    scales = [(1, 1)] * 20
    torch.manual_seed(7)
    good = views.training(captions, scales, good=True)
    bad = views.training(captions, scales)
    assert {tokens[1].item() for tokens in good} == {11, 19}
    assert len({tokens[1].item() for tokens in bad}) == 1
    pair = views.training(captions, scales, good=True, next_token=True)
    assert set(pair[:7].tolist()) == {1, 2, 11, 19}
    assert (pair[7:] == 0).all()


def test_next_token_backbone_is_causal_and_tied_head_has_gradients():
    pytest.importorskip("transformers")
    from worldlab.nn import NextTokenPrediction, TextEncoder

    encoder = TextEncoder(20, 0, arch="smoke", causal=True).eval()
    model = NextTokenPrediction(encoder)
    first = torch.tensor([[1, 4, 7, 2, 0]])
    changed = torch.tensor([[1, 4, 11, 2, 0]])
    torch.testing.assert_close(encoder.hidden(first).last_hidden_state[:, :2],
                               encoder.hidden(changed).last_hidden_state[:, :2])
    model(first).backward()
    assert encoder.bert.embeddings.word_embeddings.weight.grad.abs().sum() > 0


def test_rotation_and_retrieval_use_separate_pairs():
    rng = torch.Generator().manual_seed(9)
    a = torch.randn(40, 6, dtype=torch.float64, generator=rng)
    rotation = torch.linalg.qr(torch.randn(6, 6, dtype=a.dtype, generator=rng)).Q
    b = a @ rotation + torch.arange(6, dtype=a.dtype)
    fit, test = slice(0, 20), slice(20, 40)
    metrics, fitted = paired_alignment(a[fit], b[fit], a[test], b[test], k=4, block_size=7)
    torch.testing.assert_close(fitted(a[test]), b[test])
    assert metrics["R@1"] == metrics["mknn"] == 1.0
    assert metrics["fit_pairs"] == metrics["test_pairs"] == 20
    assert mutual_knn(a, a @ rotation, k=4, block_size=3) == 1.0
    for metric in ("cka", "mknn"):
        score, _, _ = prh_score([a], [a], k=3, metric=metric)
        assert score == pytest.approx(1.0)
    partial = Rotation.fit(a, b[:, :3])
    torch.testing.assert_close(partial.matrix.T @ partial.matrix, torch.eye(3, dtype=a.dtype))


def test_scene_fraction_is_one_for_scene_only_features():
    generator = torch.Generator().manual_seed(13)
    latent = torch.randn(30, 6, generator=generator)
    assert scene_explained_fraction(latent[:, None].expand(-1, 5, -1), 4) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="rank"):
        scene_explained_fraction(torch.zeros(30, 5, 6), 3)


def test_spectral_solution_whitens_and_satisfies_bound():
    a = torch.tensor([[4., 1., 2.], [1., 5., 1.], [2., 1., 4.]], dtype=torch.float64)
    a = a / a.sum()
    p = a.sum(1)
    b = 0.8 * a + 0.2 * torch.outer(p, p)
    embedding, eigenvalues = spectral_solution(a, 1)
    torch.testing.assert_close(p @ embedding, torch.zeros(1, dtype=a.dtype))
    torch.testing.assert_close(embedding.T @ (p[:, None] * embedding), torch.eye(1, dtype=a.dtype))
    prediction = (a * (embedding - embedding.T).square()).sum()
    torch.testing.assert_close(prediction, 2 * (1 - eigenvalues[0]))
    distance, gap, bound = mismatch_bound(a, b, 1)
    assert gap > 0 and distance <= bound


def test_join_reorders_by_id_and_uses_the_requested_split():
    pytest.importorskip("yaml")
    pytest.importorskip("torchvision")
    from worldlab.experiments.prh.evaluate import _join

    a = {"ids": ["b", "c", "a"], "splits": ["test", "val", "test"]}
    b = {"ids": ["a", "b", "c"], "splits": ["test", "test", "val"]}
    assert _join(a, b, "test") == ([2, 0], [0, 1], {"a", "b"})
    b["splits"][0] = "val"
    with pytest.raises(ValueError, match="same nonempty"):
        _join(a, b, "test")


def test_resampling_nuisance_recovers_the_scene_embedding():
    from worldlab.representation.spectral import coverage_bound

    scene = torch.tensor([[4., 1., 2.], [1., 5., 1.], [2., 1., 4.]], dtype=torch.float64)
    scene /= scene.sum()
    q = torch.tensor([0.3, 0.7], dtype=torch.float64)
    pairs = torch.kron(scene, torch.outer(q, q))
    result = coverage_bound(pairs, torch.arange(3).repeat_interleave(2), 1)
    assert result["gap"] > 0
    assert result["distance"] < 1e-12
    assert result["bound"] < 1e-12
