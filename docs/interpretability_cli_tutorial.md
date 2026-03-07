# Tutorial CLI Interpretability (B4 -> B8)

Ce tutorial rassemble les commandes importantes pour lancer les pipelines
d'interpretabilite depuis le terminal, avec des explications courtes.

## 1) Smoke test rapide

Verifie que la stack interpretability fonctionne de bout en bout.

```bash
python tools/interpretability_cli.py smoke
```

Quand l'utiliser:

- juste apres install,
- avant une campagne longue,
- en debug rapide.

## 2) Run B4 simple (real-metrics / real-attrib)

Lance un run reel depuis un fichier de config, en mode metrics only:

```bash
python tools/interpretability_cli.py real-metrics --exp-config config/interpretability/b4_example.yaml
```

Lance uniquement l'attribution:

```bash
python tools/interpretability_cli.py real-attrib --exp-config config/interpretability/b4_example.yaml
```

Quand l'utiliser:

- pour inspecter rapidement un checkpoint,
- pour verifier un dataset/config avant batch.

## 3) Run B4 batch (multi-seed / multi-checkpoint)

```bash
python tools/interpretability_cli.py real-batch --exp-config config/interpretability/b4_example.yaml --run-name batch_demo
```

Ce que ca produit:

- un dossier batch,
- `batch_runs.json` (liste des runs),
- `batch_summary.json` (agregats par checkpoint).

## 4) Run B6 matrix (grille systematique)

Lance la matrice complete definie dans la section `matrix` du YAML:

```bash
python tools/interpretability_cli.py real-matrix --exp-config config/interpretability/b4_example.yaml --run-name matrix_demo
```

Forcer un rerun complet (sans resume):

```bash
python tools/interpretability_cli.py real-matrix --exp-config config/interpretability/b4_example.yaml --run-name matrix_demo --no-resume
```

Artifacts principaux:

- `matrix_runs.json`
- `matrix_results.csv`
- `matrix_summary.json`

## 5) Run B8 publication pack (a partir d'un matrix dir)

```bash
python tools/interpretability_cli.py real-publish --matrix-dir out_interp/b4/matrix_demo
```

Version avec nommage custom:

```bash
python tools/interpretability_cli.py real-publish \
  --matrix-dir out_interp/b4/matrix_demo \
  --title "BoardGPT Interpretability Matrix" \
  --markdown-name b8_report.md \
  --notebook-name b8_report.ipynb
```

Sorties:

- rapport Markdown publication-style,
- notebook d'analyse pre-rempli.

## 6) One-shot final (B6 + B8)

Commande recommandee pour une execution complete en un seul appel:

```bash
python tools/interpretability_cli.py real-one-shot \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name matrix_pub_run
```

Sans resume (recalcule tout):

```bash
python tools/interpretability_cli.py real-one-shot \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name matrix_pub_run \
  --no-resume
```

## 7) Lire un run existant

Affiche le `summary.json` d'un run:

```bash
python tools/interpretability_cli.py report --run-dir out_interp/b4/matrix_pub_run/<run_id>
```

## 8) Dashboard Streamlit (optionnel)

```bash
streamlit run apps/interpretability_lab.py
```

Tu y retrouves:

- launcher sync/async,
- file de jobs,
- comparaison de runs,
- tags/notes persistes.

## Conseils pratiques

- Pour benchmark/repro: `validation_mode: strict` dans le YAML.
- Pour exploration UI: `validation_mode: tolerant`.
- Pour reproductibilite forte: garder `experiment.reproducibility.deterministic: true`.
- Commencer petit (`max_examples`) puis augmenter une fois la config validee.
