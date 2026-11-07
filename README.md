# WorldLab

Research code for world models, from autoregressive board-game transformers to joint-embedding predictive architectures.

- **LeJEPA** lives in `worldlab.nn`. The [reproduction guide](docs/lejepa.md) covers pretraining, linear probes, fine-tuning and experiment matrices.
- **LeWorldModel** adds action-conditioned latent dynamics and planning. See the [reproduction guide](docs/lewm.md) for training, control and physical diagnostics.
- **Platonic Representation Hypothesis** compares independently trained vision and text representations. See the [PRH experiment guide](docs/prh.md).
- **BoardGPT** lives in `worldlab.board`, with its PyTorch models in `worldlab.nn`. See the [training configuration](README_config.md) and [interpretability toolkit](README_interpretability.md).

## Run from the checkout

From the repository root, create and activate the Conda environment.

```bash
conda env create -f environment.yml
conda activate worldlab
python -m worldlab.experiments.lejepa train --preset smoke --device cpu --output runs/lejepa-smoke
```

The environment uses Python 3.12 and installs [requirements.txt](requirements.txt) for LeJEPA, LeWorldModel offline training, PRH and BoardGPT. Run the Python commands from the repository root so they can import the source directly.

If Miniconda prompts for the default Anaconda channels, run `CONDA_DEFAULT_CHANNELS='' conda env create -f environment.yml` on Linux or macOS to use conda-forge for this command.

If you already have a Conda environment, run `python -m pip install -r requirements.txt` inside it. To choose a specific CPU or CUDA build, install the matching PyTorch and torchvision wheels first, then install the requirements. Use the [PyTorch installer](https://pytorch.org/get-started/locally/) to select the wheel index and check CUDA with `python -c "import torch; print(torch.cuda.is_available())"`.

For LeWorldModel simulator evaluations, add `python -m pip install -r requirements-control.txt`.

The experiment modules are callable from the checkout.

| Workflow | Command prefix |
|---|---|
| LeJEPA | `python -m worldlab.experiments.lejepa` |
| LeWorldModel | `python -m worldlab.experiments.lewm` |
| PRH | `python -m worldlab.experiments.prh` |
| Board games | `python -m worldlab.board.games.cli` |
| BoardGPT training | `python -m worldlab.experiments.board.train` |
| Board autoencoder | `python -m worldlab.experiments.board.autoencoder` |
| Board probes | `python -m worldlab.experiments.board.probe` |

The `worldlab-*` console commands are also available after an optional editable install with `python -m pip install -e .`.

Import the board models alongside the JEPAs.

```python
from worldlab.nn import GPTConfig, GameGPT, GameAutoEncoder, LeJEPA, LeWorldModel
from worldlab.board.games import OthelloGame
```

Existing `boardGPT` imports and root training scripts are compatibility entry points. Board model state dicts and training checkpoint keys use the original format.
