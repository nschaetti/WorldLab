# Platonic representations with JEPAs

WorldLab implementation of Balestriero and LeCun, [Proving and Improving the Platonic Representation Hypothesis with JEPAs](https://arxiv.org/abs/2610.39302). The [official repository](https://github.com/galilai-group/prh-proof/tree/a2b9949fa907d4fbe562742d4b1ffa7205ab7427) provides the complete experiment registry, paper tables and advanced reproduction scripts.

## How it fits into WorldLab

- [`worldlab.nn.platonic`](../worldlab/nn/platonic.py) contains the BERT encoder, tied next-token head and `PlatonicJEPA`. The same JEPA accepts a vision or text backbone.
- [`worldlab.representation`](../worldlab/representation/) contains orthogonal alignment, rotation-free scores, scene variance estimates and finite-world spectral solutions. These tools work directly on tensors from any model.
- [`worldlab.experiments.prh`](../worldlab/experiments/prh/) wires the data, training and evaluation. It reuses WorldLab's SIGReg, optimization and runtime helpers.

The paper recipe predicts the mean of the global views from each view. It uses the identity predictor and optimizes `invariance + weight * SIGReg`. The projector sees all views together, and SIGReg sees the pooled `views × batch` observations. The original WorldLab LeJEPA preset instead applies SIGReg per view and uses a convex loss blend. The shared objective exposes both configurations.

## Train independent models

```bash
python -m pip install -e '.[prh]'
data=/datasets/flickr30k

for arm in image good-text bad-text next-token paired-next-token; do
    worldlab-prh train --preset "$arm" --data "$data" --output "runs/prh/$arm"
    worldlab-prh embed --checkpoint "runs/prh/$arm" --data "$data" \
        --layers --output "runs/prh/$arm.pt"
done
```

| Arm | Training positives |
|---|---|
| `good-text` | Each view samples from all captions of a scene, then trims and masks the chosen caption. Captions can be redrawn across views. |
| `bad-text` | All views trim and mask one caption sampled for the scene. |
| `next-token` | A causal BERT predicts the next token in one sampled caption. |
| `paired-next-token` | The same causal model reads two distinct captions from the same scene in one sequence. |
| `image` | All crops come from one sampled image or video frame. |
| `good-image` | Each crop samples its own frame from the same scene. |

Flickr30k uses the upstream Karpathy directory layout with `flickr_annotations_30k.csv` and `flickr30k-images/`. A custom JSON Lines manifest uses one row per scene. Paths are relative to the manifest or to `--root`.

```json
{"id":"clip-001","split":"train","images":["clip-001/0.jpg","clip-001/1.jpg"],"captions":["A dog crosses the grass","A black dog is running outside"]}
```

Give each scene a unique ID and use the `train`, `val` and `test` splits. A text-only arm can use a manifest with only captions, and an image-only arm can use only image paths. Training reads the selected modality and the training split. Embedding extraction uses the first caption or middle frame by default.

The default Flickr preset uses 200 epochs, AdamW, a batch of 256, 2 global and 6 local views, and SIGReg weight 0.02. Use `--set seed=1` and `--set seed=2` for the other paper seeds. Use `--set good_views=0.1` to draw good views for about 10% of training items. For a CPU smoke run, add `--smoke --device cpu` and use a small manifest. Repeat the command with `--resume` to resume an existing run at its last epoch boundary. A YAML config and repeated `--set KEY=VALUE` overrides are also supported.

## Measure alignment

```bash
worldlab-prh align runs/prh/image.pt runs/prh/good-text.pt \
    --rotation runs/prh/rotation.pt --output runs/prh/alignment.json
worldlab-prh benchmark runs/prh/image.pt runs/prh/good-text.pt --metric mknn
worldlab-prh nuisance --checkpoint runs/prh/good-text --data "$data" --dimensions 32 128
```

- `align` joins caches by scene ID. It fits a Procrustes map on validation scenes and computes mutual k-NN, R@1, R@5, whitened retrieval and cosine on test scenes. Use `--features projection` for projector outputs. The saved map can be applied with `Rotation(**torch.load(path, weights_only=True))`.
- `benchmark` measures alignment without fitting a map. It clips and normalizes each layer, then reports the maximum mutual k-NN or unbiased linear CKA over layer pairs. This is the PRH metric. For WIT-1024 or COCO with the official dataset loaders, use the upstream commands below.
- `nuisance` measures how much of the whitened representation is explained by scene identity across captions. The number of directions must fit the effective rank. Use fewer directions on smoke data.
- `theory` checks the spectral mismatch bound on random finite worlds. `spectral_solution` solves the whitened population objective, and `coverage_bound` measures how the observation kernel couples scene and nuisance directions.

```bash
worldlab-prh theory --cases 300
worldlab-prh scaling --data /path/to/prh-proof/paper/data/phase_grid.csv --fit v2
```

The scaling command fits the published functional form on the designated grid and reports held-out RMSE separately for the random and long runs. It can use the authors' CSV or a new CSV in the same format.

## Run the exact upstream recipes

The native runner targets single-device integration. It uses torchvision augmentations on CPU and evaluates a saved checkpoint on demand. The original runs used the pinned stable-pretraining scheduler and image photometric transforms on GPU, plus per-rank SIGReg in the distributed recipes. Use the upstream path for those protocols, Recap-DataComp scale, the synthetic coverage figures, captioning and unpaired Rosetta alignment.

```bash
git clone https://github.com/galilai-group/prh-proof.git
git -C prh-proof checkout a2b9949fa907d4fbe562742d4b1ffa7205ab7427
python -m pip install -e ./prh-proof
python -m pip install 'stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@ab836bf'
worldlab-prh paper-run --checkout ./prh-proof text-cap
worldlab-prh paper-run --checkout ./prh-proof --run text-cap
worldlab-prh upstream --checkout ./prh-proof evaluate -- wit-embed dinov2
```

`paper-run` reads the exact argument list for a tag from the authors' registry. It prints the command by default. `upstream` delegates to the official `train`, `evaluate`, `caption`, `figures`, `tables` or `data` entry point in the supplied checkout. For pair-free alignment, install the upstream `pairfree` extra and follow the benchmark setup in the [authors' reproduction guide](https://github.com/galilai-group/prh-proof/blob/a2b9949fa907d4fbe562742d4b1ffa7205ab7427/docs/reproduce.md).

The native smoke runs validate the workflow. Benchmark accuracy, captioning and web-scale training results require the paper datasets and compute. The native spectral checks concern the finite-world whitened problem. They do not establish uniqueness of the finite-sample neural objective with SIGReg.
