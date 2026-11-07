"""Run dataset-initialized model predictive control in the four paper environments."""

import json
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from worldlab.training.runtime import atomic_write, get_device

from .config import load_checkpoint
from .data import image_transform


def _call(method, **columns):
    return {"method": method, "args": {
        name: {"value": value, "in_dataset": isinstance(value, str)} for name, value in columns.items()
    }}


ENVIRONMENTS = {
    "pusht": ("swm/PushT-v1", {}, [
        _call("_set_state", state="state"), _call("_set_goal_state", goal_state="goal_state"),
    ]),
    "tworoom": ("swm/TwoRoom-v1", {}, [
        _call("_set_state", state="proprio"), _call("_set_goal_state", goal_state="goal_proprio"),
    ]),
    "reacher": ("swm/ReacherDMControl-v0", {"task": "qpos_match"}, [
        _call("set_state", qpos="qpos", qvel="qvel"), _call("set_target_qpos", target_qpos="goal_qpos"),
    ]),
    "cube": ("swm/OGBCube-v0", {
        "env_type": "single", "ob_type": "states", "multiview": False, "width": 224, "height": 224,
        "visualize_info": False, "terminate_at_goal": True,
    }, [
        _call("set_state", qpos="qpos", qvel="qvel"),
        _call("set_target_pos", cube_id=0, target_pos="goal_privileged_block_0_pos",
              target_quat="goal_privileged_block_0_quat"),
    ]),
}


class StableWorldModelCost(nn.Module):
    """Adapter for stable-worldmodel's batched shooting solvers.

    The paper's default policy starts each replan from a single frame.
    Latents are cached in the solver's per-plan information dictionary.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_cost(self, info, candidates):
        if "_worldlab_context" not in info:
            with torch.no_grad():
                pixels = info["pixels"][:, 0, -1:].to(candidates.device)
                goal = info["goal"][:, 0, -1:].to(candidates.device)
                info["_worldlab_context"] = self.model.encode(pixels)
                info["_worldlab_goal"] = self.model.encode(goal)[:, -1]
        return self.model.goal_cost(info["_worldlab_context"], info["_worldlab_goal"], candidates)


class _TimedSolver:
    def __init__(self, solver, device):
        self.solver = solver
        self.device = device
        self.calls = []

    @property
    def action_dim(self):
        return self.solver.action_dim

    @property
    def n_envs(self):
        return self.solver.n_envs

    @property
    def horizon(self):
        return self.solver.horizon

    def configure(self, **kwargs):
        self.solver.configure(**kwargs)

    def __call__(self, info, **kwargs):
        return self.solve(info, **kwargs)

    def solve(self, info, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = self.solver.solve(info, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.calls.append({"seconds": time.perf_counter() - start, "environments": len(info["pixels"])})
        return result


def evaluation_starts(lengths, offset, count, seed, *, reference_sampling=True):
    """Sample uniform valid rows. The reference excludes the last eligible row."""
    if min(offset, count) < 1:
        raise ValueError("Goal offset and evaluation count must be positive")
    counts = np.maximum(np.asarray(lengths, dtype=np.int64) - offset, 0)
    ends = counts.cumsum()
    available = (int(ends[-1]) if len(ends) else 0) - int(reference_sampling)
    if available < count:
        raise ValueError(f"Requested {count} starts, only {max(0, available)} are available")
    samples = np.sort(np.random.default_rng(seed).choice(available, size=count, replace=False))
    episodes = np.searchsorted(ends, samples, side="right")
    starts = samples - np.concatenate(([0], ends))[episodes]
    return episodes.tolist(), starts.tolist()


def evaluate_control(args):
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm

    device = get_device(args.device)
    if min(args.horizon, args.receding, args.frame_skip, args.budget, args.batch_size) < 1:
        raise ValueError("Planning and budget sizes must be positive")
    if args.receding > args.horizon or args.horizon * args.frame_skip > args.budget:
        raise ValueError("Require receding <= horizon and horizon * frame_skip <= budget")
    env_name, options, specs = ENVIRONMENTS[args.environment]
    callables = deepcopy(specs)
    keys = {"pixels", "action"}
    path = Path(args.data).expanduser()
    with h5py.File(path, "r") as source:
        for call in callables:
            for spec in call["args"].values():
                if not spec["in_dataset"]:
                    continue
                key = spec["value"].removeprefix("goal_")
                if key not in source and key.startswith("privileged_"):
                    alternative = key.replace("privileged_", "privileged/", 1)
                    if alternative in source:
                        spec["value"] = spec["value"].replace(key, alternative)
                        key = alternative
                keys.add(key)
        if "seed" in source:
            keys.add("seed")
    dataset = swm.data.HDF5Dataset(path=path, keys_to_load=sorted(keys), keys_to_cache=["action"])
    episodes, starts = evaluation_starts(dataset.lengths, args.goal_offset, args.episodes, args.seed)
    iterations = args.iterations if args.iterations is not None else (30 if args.environment == "pusht" else 10)
    samples = args.samples if args.samples is not None else (300 if args.solver == "cem" else 100)
    timing = None
    if args.solver == "random":
        policy = swm.policy.RandomPolicy(seed=args.seed)
    else:
        if not args.checkpoint:
            raise ValueError("LeWM control requires --checkpoint")
        model, resolution, action_stats = load_checkpoint(args.checkpoint, device)
        model.requires_grad_(False)
        raw = np.asarray(dataset.get_col_data("action"))
        raw = raw.reshape(len(raw), -1)
        finite = raw[np.isfinite(raw).all(axis=1)]
        if model.action_encoder.patch_embed.in_channels != args.frame_skip * raw.shape[1]:
            raise ValueError("Checkpoint action width differs from environment action width * frame_skip")
        scaler = StandardScaler().fit(finite)
        if action_stats is not None:
            scaler.mean_ = np.asarray(action_stats["mean"])
            scaler.scale_ = np.asarray(action_stats["std"])
        planner = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding, action_block=args.frame_skip)
        cost = StableWorldModelCost(model)
        if min(iterations, samples) < 1:
            raise ValueError("Solver iterations and sample count must be positive")
        common = dict(model=cost, batch_size=args.batch_size, num_samples=samples, n_steps=iterations,
                      device=str(device), seed=args.seed)
        if args.solver == "cem":
            if not 2 <= args.elites <= samples:
                raise ValueError("CEM requires 2 <= elites <= samples")
            solver = swm.solver.CEMSolver(**common, topk=args.elites, var_scale=1.0)
        else:
            optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                          "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
            solver = swm.solver.GradientSolver(**common, action_noise=0,
                                              optimizer_cls=optimizers[args.solver],
                                              optimizer_kwargs={"lr": args.solver_lr})
        timing = _TimedSolver(solver, device)
        transform = image_transform(resolution)
        policy = swm.policy.WorldModelPolicy(
            solver=timing, config=planner, process={"action": scaler},
            transform={"pixels": transform, "goal": transform},
        )
    world = swm.World(env_name=env_name, num_envs=args.episodes, image_shape=(224, 224),
                      max_episode_steps=2 * args.budget, **options)
    try:
        world.set_policy(policy)
        if "seed" not in keys:
            world.reset(seed=args.seed)
        started = time.perf_counter()
        result = world.evaluate(dataset=dataset, episodes_idx=episodes, start_steps=starts,
                                goal_offset=args.goal_offset, eval_budget=args.budget,
                                callables=callables, seed=args.seed, video=args.video)
        elapsed = time.perf_counter() - started
    finally:
        world.close()
    record = {"config": vars(args), "solver_iterations": iterations, "solver_samples": samples,
              "episodes": episodes, "starts": starts, "seconds": elapsed, "metrics": result}
    if timing is not None:
        record["planning_calls"] = timing.calls
        record["planning_seconds"] = sum(call["seconds"] for call in timing.calls)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(destination, lambda file: file.write_text(
        json.dumps(record, default=lambda value: value.tolist(), indent=2) + "\n",
    ))
    print(json.dumps({"success_rate": float(result["success_rate"]), "seconds": elapsed}), flush=True)
    return record
