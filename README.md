# ColBERT-PPI

Protein partner retrieval with reusable residue-level representations. ColBERT-PPI compares proteins through multi-vector matching and supports adaptation to protein–RNA retrieval.

## Quick start — CPU, no model download

```bash
git clone https://github.com/UR-Free/ColBERT-PPI.git
cd ColBERT-PPI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 python src/run_example.py
```

Expected output includes `"status": "PASS"` and maximum matrix errors below
`2e-6` and a score near `-0.01595231`. This runs a real cached residue-vector example; it verifies scoring
without neural encoding or retraining. Linux and Python 3.10 are the reference
neural environment. The core CPU example also supports Python 3.12.

## Predict with the released PPI model

```bash
python -m pip install -e '.[ppi]'
python scripts/download_assets.py ppi
```

Download the upstream [SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB)
Hugging Face model files into `data/weights/backbones/SaProt_650M_PDB/`.
The directory must contain `pytorch_model.bin`, `config.json`, `vocab.txt`,
`tokenizer_config.json`, and `special_tokens_map.json`. A standalone `.pt`
file is not a substitute for this directory. Then run:

```bash
DEVICE=cpu bash PPI_inference.sh
# For CUDA:
DEVICE=cuda:0 bash PPI_inference.sh
```

Predictions are written to `data/predictions/ppi.json` as pair IDs and scores.
Higher scores rank candidate partners; scores are **not probabilities** and
no universal interaction threshold is supplied. The checkpoint and its
reference bank must stay together.

To score your own proteins, follow [PDB input preparation](docs/INPUTS.md).
See [model files and manual downloads](docs/WEIGHTS.md), the
[model card](MODEL_CARD.md), and [environment troubleshooting](docs/ENVIRONMENT.md).
Large assets are hosted in the [preprint release](https://github.com/UR-Free/ColBERT-PPI/releases/tag/v0.2.0-preprint);
the downloader verifies their SHA-256 hashes from `assets.json`.

## Optional protein–RNA (PRI) workflow

PRI additionally needs the upstream [ERNIE-RNA source and pretrained weights](https://github.com/Bruce-ywj/ERNIE-RNA),
and a Python 3.10 environment with its legacy dependencies:

```bash
python -m pip install 'pip<24.1'
python -m pip install -r requirements-pri.txt
python scripts/download_assets.py pri
bash PRI_inference.sh
```

Set `ERNIE_CODE`, `ERNIE_CHECKPOINT` and `SAPROT_DIR` in the environment or
`config/PRI_inference.env`. Use `DEVICE=cpu` when CUDA is unavailable.
The supplied PRI input is already tokenized; raw RNA preprocessing is not
included. PRI training with the default PPI initialization also needs the PPI asset.

All four shell entry points read their matching files in `config/`.
Environment variables override defaults; `PYTHON` selects the interpreter.
Use `--dry-run` to inspect a command. Relative paths resolve from the repository
root. Direct Python commands should be run from that directory.

## Training

```bash
bash PPI_train.sh
bash PRI_train.sh
```

Defaults use ten training pairs and one epoch per task. Validation AUPRC selects the checkpoint, which is then evaluated on a separate test split. Outputs are saved to `data/runs/ppi_demo/` and `data/runs/pri_demo/`. These small datasets demonstrate training and evaluation; use the benchmark datasets below to inspect the reference retrieval results.

PRI training initializes the protein branch from the PPI checkpoint specified by `PROTEIN_INIT`. Set it to an empty value for SaProt-only initialization:

```bash
PROTEIN_INIT="" bash PRI_train.sh
DEVICE=cuda:1 EPOCHS=2 bash PPI_train.sh
bash PPI_train.sh --epochs 2 --batch-size 1
```

Use `--dry-run` to print a command without loading models. See [training details](docs/TRAINING.md) for objectives and output files.

## Pair inference

```bash
bash PPI_inference.sh
bash PRI_inference.sh
```

The default JSON inputs contain tokenized pairs without contact labels. Outputs are written to `data/predictions/ppi.json` and `data/predictions/pri.json`. Set `INPUT`, `CHECKPOINT` and `OUTPUT` to use other inputs. PPI also requires the checkpoint's reference bank.

Pair inference uses the complete multi-vector models. Use benchmark mode for sequence-only and single-vector controls. Input fields and token conventions are described in [DATA_GUIDE.md](docs/DATA_GUIDE.md).

## Benchmark evaluation

Download the prepared inputs, frozen labels and numerical evidence first:

```bash
python scripts/download_assets.py benchmarks
python -m pip install -e '.[ppi,analysis]'
```


```bash
bash PPI_inference.sh --benchmark --split test
bash PRI_inference.sh --benchmark --split test
```

Use `--split validation` for the validation inputs. Benchmark mode reads the checkpoint's adjacent `config.json` to select multi-vector, sequence-only or single-vector scoring. Set `CHECKPOINT` to choose a model and `BENCHMARK_OUTPUT` to separate output directories. Defaults are `data/benchmarks/ppi/` and `data/benchmarks/pri/`.

For example, evaluate the single-vector PPI control:

```bash
CHECKPOINT=data/weights/ppi/ppi_single/weights.pt \
BENCHMARK_OUTPUT=data/benchmarks/ppi_single \
bash PPI_inference.sh --benchmark --split test
```

Run the independent Y2H screen with:

```bash
python src/benchmark.py --task y2h \
  --checkpoint data/weights/ppi/ppi_full/weights.pt \
  --saprot-dir data/weights/backbones/SaProt_650M_PDB \
  --output data/benchmarks/y2h
```

The benchmark runner uses the supplied labels and cohort masks. Repeated protein accessions are collapsed for PPI; exact protein/RNA groups are collapsed for PRI. Duplicate scores use the maximum, and positive labels take precedence.

| Readout | Scoring |
|---|---|
| Multi-vector PPI and Y2H | Reference calibration, α = 1, τ = 0.03 |
| Multi-vector PRI | Smooth MaxSim, α = 0, τ = 0.001 |
| Single-vector controls | Attention-weighted pooled cosine with the model's scale |
| Interface localisation | Raw residue cosine matrix, averaged over encoder roles |

## Reproduce metrics and figures

Install `.[analysis]` and download the `benchmarks` asset for these commands.
Figure SVG export additionally needs the system Cairo library.


The following CPU commands use saved vectors or predictions:

```bash
python src/run_example.py
python src/reproduce_results.py
```

The first checks scoring on one cached protein pair. The second recalculates retrieval and localisation metrics and writes results to `data/validation_reports/`. Neither requires model downloads.

```bash
python src/plot_figures.py
python src/plot_figures.py --official
```

The first command replots numerical evidence from source tables. `--official` exports the final SVG compositions in `data/figure_templates/`, including schematic and structural assets. Template export preserves the composition and does not recalculate its data.

## Repository layout

```text
PPI_train.sh / PRI_train.sh
PPI_inference.sh / PRI_inference.sh
src/       Models, scoring, training and evaluation
config/    Launch configurations and model settings
data/      Inputs, labels, weights, source tables and figure templates
docs/      Data formats, environment and method details
```

[CONTENTS.md](docs/CONTENTS.md) describes the included datasets and model components. [CODE_GUIDE.md](docs/CODE_GUIDE.md) maps the implementation. Usage terms are in [LICENSE](LICENSE). Third-party model and data terms are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Scope, citation and support

This release follows the September 2026 manuscript package. Its PPI readout
uses reference calibration and smooth MaxSim; it supersedes the historical
GitHub top-k/top-N prototype. See [changes](docs/CHANGELOG.md).

Full original training data and its end-to-end preprocessing are not included.
The small training splits demonstrate optimization and validation selection;
they do not reproduce the manuscript training experiment. Supplied benchmark
labels and frozen predictions support evaluation and numerical reproduction.

Use [CITATION.cff](CITATION.cff) to cite the software and include the exact
commit or release tag. A preprint DOI will be added when available.
For questions or reproducible errors, open a
[GitHub issue](https://github.com/UR-Free/ColBERT-PPI/issues) with the command,
Python/package versions and traceback.

Development checks: `pip install -e '.[dev]'`, then `pytest -q` and
`python -m build`. A [CI template](docs/ci/cpu.yml) is provided; activation
instructions and tested scope are in [validation](docs/VALIDATION.md).
