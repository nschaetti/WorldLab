"""CPU checks of the spectral bound and fits of the empirical alignment law."""

import csv

import numpy as np
import torch

from worldlab.representation.spectral import mismatch_bound

from .evaluate import _json


def _positive_pairs(affinity, marginal):
    """Symmetrically balance a positive affinity to a common marginal."""
    diagonal = (marginal / affinity.sum(1)).sqrt()
    for _ in range(10000):
        product = affinity @ diagonal
        if (diagonal * product - marginal).abs().max() < 1e-13:
            break
        diagonal = (diagonal * marginal / product).sqrt()
    else:
        raise RuntimeError("Positive-pair balancing did not converge")
    return diagonal[:, None] * affinity * diagonal[None, :]


def check_theory(args):
    if args.cases < 1:
        raise ValueError("Number of cases must be positive")
    rng = np.random.default_rng(args.seed)
    ratios = []
    for index in range(args.cases):
        states, k = int(rng.integers(8, 40)), int(rng.integers(1, 6))
        marginal = torch.from_numpy(rng.dirichlet(np.ones(states)))

        def affinity():
            matrix = torch.from_numpy(rng.random((states, states)))
            return matrix + matrix.T

        original = affinity()
        if index % 2:
            labels = torch.from_numpy(rng.integers(0, 3, size=states))
            original[labels[:, None] != labels[None, :]] *= 10 ** -rng.uniform(0, 3)
        reference = _positive_pairs(original, marginal)
        scale = 10 ** -rng.uniform(0, 4)
        perturbed = (1 - scale) * reference + scale * _positive_pairs(affinity(), marginal)
        distance, gap, bound = mismatch_bound(perturbed, reference, k)
        if gap <= 0:
            raise ValueError("The sampled world has no spectral gap")
        ratios.append(distance / bound)
    result = {"cases": args.cases, "seed": args.seed, "max_distance_over_bound": max(ratios),
              "violations": sum(ratio > 1 + 1e-8 for ratio in ratios)}
    _json(args.output, result)
    return result


def scaling_law(x, a, d, b, c, alpha, beta, gamma):
    """R@1 from good-view fraction q, samples M and distinct scenes N."""
    q, samples, scenes = x
    return a + d * np.log(samples) + b * np.log1p(q**alpha * samples**beta * scenes**gamma / c)


def fit_scaling(args):
    from scipy.optimize import curve_fit

    with open(args.data, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fit_rows = [row for row in rows if row["design"] == "grid"
                and (args.fit == "v2" or float(row["samples_M"]) <= 20)]
    held_out = [row for row in rows if row not in fit_rows]
    if len(fit_rows) < 7:
        raise ValueError("At least seven grid runs are required for the seven-parameter law")

    def arrays(records):
        return np.array([[float(row[key]) for row in records] for key in ("q", "samples_M", "images_M", "r1")])

    def rmse(x, y, fitted):
        return float(np.sqrt(np.mean((y - scaling_law(x, *fitted)) ** 2)))

    data = arrays(fit_rows)
    fitted, _ = curve_fit(scaling_law, data[:3], data[3], p0=[0.2, 0.02, 0.03, 1, 1, 1, 0.1],
                          bounds=([0, -1, 0, 0.001, 0.1, 0.1, -1], [1, 1, 1, 100, 5, 5, 2]),
                          maxfev=20000)
    result = {"fit": args.fit, "rows": len(fit_rows),
              "parameters": dict(zip(("a", "d", "b", "c", "alpha", "beta", "gamma"), fitted.tolist())),
              "fit_rmse": rmse(data[:3], data[3], fitted)}
    for design in sorted({row["design"] for row in held_out}):
        test = arrays([row for row in held_out if row["design"] == design])
        result[f"heldout_{design}_rmse"] = rmse(test[:3], test[3], fitted)
    _json(args.output, result)
    return result
