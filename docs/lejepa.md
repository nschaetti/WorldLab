# LeJEPA in WorldLab

Implementation of Balestriero and LeCun, [arXiv:2511.08544v3](https://arxiv.org/abs/2511.08544v3), Algorithms 1–2. The [official implementation](https://github.com/galilai-group/lejepa/tree/c293d29) provides a runnable Imagenette example in `MINIMAL.md`, a statistics library and several launch grids. The launch grids refer to a training script and configs absent from that revision. WorldLab implements the published method in PyTorch and records the choices needed to run those grids.

## Installation and first runs

Install a PyTorch build for your hardware, then install the experiment dependencies.

```bash
python -m pip install -e '.[lejepa,imagenette]'
OMP_NUM_THREADS=2 worldlab-lejepa train --preset smoke --device cpu --output runs/smoke
worldlab-lejepa train --preset imagenette-minimal --set data_root=data/lejepa/imagenette --output runs/imagenette
```

The smoke run uses torchvision FakeData. The Imagenette recipe follows the official example with 128 px crops, ViT-S/8, 4 views, projection dimension 16, 256 slices, 800 epochs and λ = 0.02. Set `--set download=false` to use a prepared ImageFolder dataset. The paper's ImageNet-10 benchmark uses a larger subset than the torchvision Imagenette recipe. Supply the paper's split as an ImageFolder when available.

Use `--config file.yaml` to load overrides on top of a preset and repeat `--set key=value` for individual overrides. Resolved settings and package versions go into `run.json`, epoch records into `metrics.jsonl`, and resumable state into `last.pt`. Resume with the same command and add `--resume`. Resumption is at epoch boundaries with the same configuration and world size. Metrics are fractions in `[0, 1]`.

```bash
torchrun --standalone --nproc-per-node=4 -m worldlab.experiments.lejepa train \
    --preset inet1k-vit-large --set data_root=/datasets/imagenet1k --output runs/vit-large
```

`batch_size` is global. It must divide by the number of ranks with at least two samples per rank. SIGReg computes the empirical characteristic function from samples across all ranks. For global BatchNorm statistics on CUDA, set `sync_batchnorm=true`. ViT activation checkpointing and SIGReg slice chunking trade recomputation for memory. Adjust `batch_size`, crop sizes or backbone for a smaller run and record the change when comparing results.

## Datasets

| Dataset key | Layout or source |
|---|---|
| `inet1k`, `inet100`, `galaxy10`, `imagefolder` | ImageFolder with `train/<class>/...` and `val/<class>/...`. Both splits must have the same class names. |
| `imagenette` | Local ImageFolder or torchvision `Imagenette` (`160px` variant) with `--set download=true`. |
| `cifar10`, `cifar100`, `flowers102`, `food101`, `dtd`, `aircraft`, `pets` | Torchvision splits with optional download. Local ImageFolder splits also work. |
| `cars` | Prepared ImageFolder recommended. |
| `fake` | Tiny generated RGB dataset for smoke tests. |

For matrices, `--data-root` is the common parent with one subdirectory per dataset key. For `train` and `probe`, use the specific dataset directory. Keep the exact class list and split files with any published runs, especially for ImageNet-100, ImageNet-10 and Galaxy10.

## Pretraining and evaluation

```bash
worldlab-lejepa matrix table7-lambda-views --data-root /datasets --output runs
worldlab-lejepa matrix table7-lambda-views --data-root /datasets --output runs --index 0
worldlab-lejepa summarize --runs runs --output results/pretraining.csv

worldlab-lejepa probe --checkpoint runs/imagenette/last.pt \
    --dataset cifar10 --data-root /datasets/cifar10 --download \
    --shots 10 --seed 0 --mode linear --output runs/probe-cifar10-10
```

Without `--index`, a matrix prints an indexed JSON Lines manifest. Each index is an independent job for a cluster array. Use `torchrun … -m worldlab.experiments.lejepa matrix … --index N` for a multi-GPU pretraining job. `probe` uses one device. It can cache frozen features and fit a normalized linear probe or fine-tune the encoder with `--mode finetune`. Few-shot selection samples up to the requested examples per class from the training split. Validation always uses the full held-out split.

| Paper result | Entry point | Protocol status |
|---|---|---|
| Table 1 (a–e) | `matrix table1-epps`, `table1-views`, `table1-batch`, `table1-projection`, `table1-registers` | Published table cells, with resolved settings described below. |
| Fig. 8 / Table 7 | `matrix table7-lambda-views` | ResNet-50 on ImageNet-100, with the published view counts and λ values. |
| Fig. 9 | `matrix figure9-architectures` | Fixed list of 50 timm backbones, including the labeled architectures supported by timm. The complete author manifest is unavailable. |
| Figs. 10–11, 19 and 21 | `matrix figure11-correlation`, then `correlate` | Loss components and rank-correlation data. The hyperparameter grid and epoch aggregation are specified in WorldLab. |
| Fig. 12 / Table 3 | `transfer-matrix figure12-galaxy10` | Frozen and fine-tuned runs across label budgets, with available timm baselines. |
| Table 4 | `matrix table4-heuristics` | Projector depth, predictor and SWA toggle. The SWA teacher updates at each epoch boundary. |
| Table 5 | `matrix table5-in-domain` | In-domain backbones and datasets. Use `probe` for the frozen evaluation and external baseline. |
| Fig. 1, §6.4 and Table 2 | `matrix imagenet-scale`, then `transfer-matrix table2-transfer` | ViT-L/14, ConvNeXtV2-H and ViT-gigantic. Table 2 includes both LeJEPA encoders and available I-JEPA checkpoints. |
| Figs. 13–14 | `visualize` | Per-image patch PCA, attention and masks. Pass ordered frames for frame-by-frame analysis. |
| Figs. 6–7, 16 and 20 / Table 6 | `synthetic`, `quadrature`, `timing` | Synthetic tests, numerical integration accuracy and SIGReg timing. |

For the Galaxy10 transfer matrix, pretrain the four LeJEPA encoders first. For example, use `train --preset in-domain --set backbone=resnet18 --set data_root=/datasets/galaxy10 --output runs/galaxy10/resnet18` and repeat with `convnextv2_nano`, `levit_128` and `resnet34`. The transfer matrix looks for these checkpoints under `--pretrained-runs runs/galaxy10`.

```bash
worldlab-lejepa transfer-matrix table2-transfer --pretrained-runs runs \
    --data-root /datasets --output runs/transfer
worldlab-lejepa summarize-probes --runs runs/transfer --output results/transfer.csv
worldlab-lejepa correlate --runs runs/figure11-correlation --output results/correlation.csv
worldlab-lejepa quadrature --output results/quadrature.csv
```

### Settings and limits

- Algorithm 2 uses the mean squared error to the global-view centroid and SIGReg on each view. The centroid is differentiable in the default recipe. Table 1 uses the SWA teacher to follow the public launch files. Table 4 tests both modes.
- The ImageNet ViT presets use 238 px global crops, 98 px local crops and 30% patch dropout, as in the public ablation launch files. The v3 article gives 224/96 in its general recipe. Non-ViTs default to same-resolution views with all views global. The main presets use the public augmentation and optimizer recommendations, with one warmup epoch and a cosine floor at one thousandth of each group's peak LR. The minimal Imagenette recipe uses the example's absolute LR floor instead.
- The default Epps–Pulley implementation evaluates the even integrand on 17 positive-half grid points and integrates over the full symmetric range. Set `symmetric_grid=true` to count the supplied integration points across the whole range as in Algorithm 1. Choose fixed directions with an explicit `step`, or pass your own directions for controlled experiments.
- The matrices save the online probe metric. Frozen, independently trained probes are separate jobs. The public material does not specify every downstream setting, split, random seed, teacher update rule, or final model selection policy. The exact I-JEPA + STOP checkpoint and the paper's video inputs are required to duplicate those comparisons. The auxiliary theory illustrations and original figure styling are outside these scripts.
- The GPU accuracy and V100 timings in the paper have not been rerun. Report any experimental deviations with the resolved config and dataset provenance.

## Reuse with other modalities

```python
import torch
from torch import nn

from worldlab.nn import LeJEPA, LeJEPAObjective, ProjectionMLP, SIGReg

model = LeJEPA(
    encoder=nn.Linear(64, 128),
    projector=ProjectionMLP(128, 32, hidden_dims=(256,)),
    objective=LeJEPAObjective(weight=0.05, sigreg=SIGReg(num_slices=256)),
)
views = torch.randn(4, 16, 64)  # views, batch, input features
result = model(views, num_global=2, step=0)
result.loss.backward()
```

Pass an encoder that produces one vector per sample. For already computed projections, call `LeJEPAObjective` directly with shape `(views, batch, features)`. The first `num_global` views form the target. In distributed training, pass the same optimizer step on every rank.
