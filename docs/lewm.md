# LeWorldModel in WorldLab

This implementation follows Maes et al., [LeWorldModel v3](https://arxiv.org/abs/2603.19312v3), and the [authors' code at 8edfeb3](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). The model, causal action-conditioned predictor and action embedder live in `worldlab.nn`. LeWM reuses LeJEPA's SIGReg and the shared training utilities. The trajectory optimizers in `worldlab.planning` accept a batched action cost and can also be used outside the simulator integration.

## Install and prepare data

The [Conda environment](../README.md#run-from-the-checkout) covers offline training and diagnostics. From the repository root, add the simulator dependencies for control evaluations. This integration uses stable-worldmodel 0.1.1.

```bash
python -m pip install -r requirements-control.txt
```

Download the HDF5 archives from the [authors' Hugging Face collection](https://huggingface.co/collections/quentinll/lewm-69c58f5e20580628114d20c4). Decompress the Push-T HDF5 with `zstd -d` and the `.tar.zst` archives with `tar --zstd -xf`. For the matrices, organize them under the shared data root as follows. Direct commands accept any valid path.

| Preset | Path relative to data root | Paper training context |
|---|---|---:|
| `pusht` | `pusht_expert_train.h5` | 3 |
| `tworoom` | `tworoom.h5` | 1 |
| `cube` | `ogbench/cube_single_expert.h5` | 3 |
| `reacher` | `dmc/reacher_random.h5` | 3 |

The loader expects `pixels`, `action`, `ep_len` and `ep_offset`. It reads only complete clips within an episode. Each action block contains the consecutive controls leaving its paired frame. The last frame is a prediction target. SIGReg applies to all frames, separately for each time index across the batch. Control also needs the state fields used by the environment to set the initial and goal states.

## Train and evaluate

```bash
python -m worldlab.experiments.lewm train --preset pusht --data /datasets/lewm/pusht_expert_train.h5 \
    --device cuda --output runs/lewm/pusht
python -m worldlab.experiments.lewm control --environment pusht --data /datasets/lewm/pusht_expert_train.h5 \
    --checkpoint runs/lewm/pusht --device cuda --output runs/lewm/pusht/control.json
```

Training saves `run.json`, `metrics.jsonl` and `last.pt`. Repeat the training command with `--resume` to resume at an epoch boundary. Override a preset with `--config file.yaml` or `--set KEY=VALUE`. The runner uses a single device, as in the reference recipe. `--preset smoke` uses a small CNN for development on a small HDF5 file.

The control protocol samples 50 initial states, selects goals 25 environment steps ahead and allows 50 actions per episode. CEM uses a five-block horizon, five actions per block, 300 candidates and 30 elites. The v3 appendix specifies 30 iterations on Push-T and 10 on the other environments. Set `--iterations 30` to match the public evaluation YAML on those environments. The JSON reports individual outcomes, sampled starts, total evaluation time and timing for each solver call. Use `--solver random` for the random-policy baseline or `--solver adam|adamw|sgd|rmsprop` for gradient-based planning.

Released checkpoints load directly from a directory containing the model repo's `config.json` and `weights.pt`. No pickle object checkpoint or upstream model import is needed. For example, run `hf download quentinll/lewm-pusht --local-dir checkpoints/pusht` and pass `--checkpoint checkpoints/pusht`. Model repo suffixes are `pusht`, `tworooms`, `cube` and `reacher`. The released TwoRoom checkpoint encodes a predictor context of three. The training preset follows the v3 appendix and uses one.

## Paper runs

```bash
python -m worldlab.experiments.lewm matrix main --data-root /datasets/lewm --output runs/lewm
python -m worldlab.experiments.lewm matrix main --data-root /datasets/lewm --output runs/lewm --index 0 --run train
python -m worldlab.experiments.lewm matrix main --data-root /datasets/lewm --output runs/lewm --index 0 --run control
```

| Result | WorldLab entry point |
|---|---|
| Fig. 6 control | `matrix main`, or `control` with a released checkpoint |
| Fig. 3 planning time and compute | Timing in `control`; `matrix compute` varies CEM samples and iterations |
| Figs. 7, 9, 10 and 17 | `decoder` trains a frozen-encoder decoder. `rollout` reports error by horizon and latent straightness, and can export an image strip using the decoder. |
| Tables 1, 3 and 4 / Fig. 13 | `probe` fits a linear probe and an MLP. Optional `--tsne POINTS` exports coordinates and physical variables. |
| Figs. 8, 11, 12 and 14 | `surprise` scores aligned reference, recoloring and teleportation clips, and computes paired tests at the perturbation step. |
| Figs. 15–16 | `matrix sigreg` varies latent dimension, slices, integration knots and regularization weight. |
| Tables 5–9 | `matrix variance` and `matrix architecture` cover training seeds, predictor depth, encoder choice, dropout and joint reconstruction. |
| Table 10 | `matrix solver` evaluates CEM and gradient solvers against the three Push-T checkpoints from the variance suite. |
| Fig. 18 | Per-epoch losses in `metrics.jsonl`. |

```bash
python -m worldlab.experiments.lewm probe --checkpoint checkpoints/pusht --data /datasets/lewm/pusht_expert_train.h5 \
    --target agent=state:0,1 --target block=state:2,3 --target angle=state:4 \
    --tsne 1000 --output results/pusht-probes

python -m worldlab.experiments.lewm surprise --checkpoint checkpoints/pusht --data reference.h5 \
    --visual recolored.h5 --physical teleported.h5 --perturbation-step 50 \
    --stats-data /datasets/lewm/pusht_expert_train.h5 --output results/pusht-surprise
```

### Protocol choices and remaining checks

- Training defaults to the paper's 10 epochs and uses the public optimizer and loss settings, including λ = 0.09, 17 knots on the positive half of `[-3,3]`, and 1,024 directions. WorldLab applies warmup over the first 1% of optimizer steps and then cosine decay. The public YAML instead sets 100 epochs and declares an epoch-level scheduler.
- The default 90/10 window split matches the public training approach and can include overlapping windows in both splits. Use `--set split=episodes` for an episode-disjoint evaluation. Physical probes use an episode split and normalize each target using the training partition. Probe, decoder and ablation settings that are unavailable in the release are explicit in the runner or matrix.
- The Push-T control adapter uses the sampled future state as its goal, which matches the paper's protocol. The reference YAML passes the initial state to the goal setter. The released solver stack plans from the current frame at each replan. The native model can also roll out from multiple context frames when their executed action blocks are available.
- The full GPU retraining, 50-episode scores on the released datasets, and comparative speedup have yet to be verified. The compute suite is a CEM budget sweep. Equal-FLOP comparison requires the corresponding baseline models and a shared profiler. The baseline training pipelines and the authors' exact recoloring and teleportation trajectory generators are outside this integration. Supply matched HDF5 trajectories for surprise scoring.

## Reuse the model

```python
from worldlab.experiments.lewm.config import load_checkpoint

model, resolution, action_stats = load_checkpoint("checkpoints/pusht")
# frames: normalized (batch, context, 3, height, width)
context = model.encode(frames)
# future_actions: normalized (batch, candidates, horizon, blocked_action_dim)
predictions = model.rollout(context, future_actions, past_actions)
```

`past_actions` contains the executed action blocks between context frames and can be omitted for a single-frame context. For training with a custom encoder, instantiate `worldlab.nn.LeWorldModel` and call it with a batch of `history + 1` frames and the aligned action blocks.
