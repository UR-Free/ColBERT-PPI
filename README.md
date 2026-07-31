# ColBERT-PPI

Structure-aware, residue-explicit protein-pair modelling with SaProt and late
interaction.

[Private repository](https://github.com/UR-Free/ColBERT-PPI) ·
[Data format](docs/DATA_FORMAT.md) ·
[Development decisions](docs/DEVELOPMENT_DECISIONS.md) ·
[Release audit](docs/RELEASE_AUDIT.md)

## What this repository provides

ColBERT-PPI encodes two proteins independently from SaProt amino-acid/3Di
tokens, compares their contextualized residue embeddings, and uses the
resulting interaction matrix for two related tasks:

```text
SaProt tokens → independent protein encoders → residue embeddings
                                                │
                                                ├─ contact supervision
                                                └─ protein-pair scoring
```

The publication code has a deliberately narrow scope:

| Component | Publication setting |
|---|---|
| Training objective | Sampled bidirectional residue-contact InfoNCE only |
| Pair representation | Residue-level late interaction |
| Canonical PPI score | Explicit role swap, mutual top-1, top-10 sum, orientation mean |
| Checkpoint selection | Validation data only |
| Bundled data or weights | None |

Protein-level InfoNCE, AUC surrogate losses, attention regularization,
self-hard-negative loss and pooled single-vector scoring are not present.
AUPRC, AUROC, MRR and Hit@K are evaluation metrics only.

## Repository layout

```text
.
├── configs/
│   └── contact_only_example.json    # reproducible 80-epoch template
├── docs/
│   ├── DATA_FORMAT.md               # required local artifact schemas
│   ├── DEVELOPMENT_DECISIONS.md     # binding implementation decisions
│   └── RELEASE_AUDIT.md             # privacy and release checks
├── src/colbert_ppi/
│   ├── adapters/                    # optional LM and LoRA adapters
│   ├── cli/                         # installed training/evaluation commands
│   ├── config.py                    # training defaults
│   ├── dataset.py                   # PPI datasets and batch samplers
│   ├── model.py                     # SaProt and late-interaction model
│   ├── protein_nucleic.py           # separate protein–nucleic task
│   ├── retrieval.py                 # read-only retrieval scores and metrics
│   ├── scoring.py                   # frozen canonical PPI scorer
│   └── trainer.py                   # contact-only train/eval loops
├── tests/
│   └── test_contact_only_loss.py
└── pyproject.toml
```

The project uses a standard `src` layout. Import reusable functionality from
`colbert_ppi`; use the installed commands for experiments.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
git clone https://github.com/UR-Free/ColBERT-PPI.git
cd ColBERT-PPI

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

SaProt weights are not redistributed. Download the required checkpoint under
its original licence and set `saprot_dir` in the experiment configuration.

## Prepare local data

No biological records are bundled. Training requires:

1. paired protein identifiers in CSV format;
2. pre-tokenized SaProt inputs keyed by protein identifier;
3. Cβ coordinates keyed by protein identifier;
4. sparse positive and negative contact labels keyed by pair identifier.

Keep these artifacts outside version control. Their exact schemas are defined
in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Train

Create a local configuration from the provided template:

```bash
cp configs/contact_only_example.json configs/local.json
```

Edit its data and SaProt paths, then start a single-GPU run:

```bash
colbert-ppi-train --config configs/local.json --gpu 0
```

For two GPUs:

```bash
colbert-ppi-train --config configs/local.json --gpu 0,1
```

The template uses 80 epochs, validation-only checkpoint selection and no
per-epoch test evaluation. Outputs are written beneath `outputs/`, which is
ignored by Git.

## Evaluate a frozen checkpoint

```bash
colbert-ppi-evaluate \
  --run-dir outputs/run_YYYYMMDD_HHMMSS \
  --checkpoint best_model.pt \
  --split test \
  --labels data/processed/test_labels.npz \
  --output-dir results/test
```

The evaluator writes `scores.npz` and `metrics.json`. It explicitly evaluates
both role assignments, applies mutual row/column top-`k=1` filtering, sums the
largest `N=10` retained similarities and averages the two orientation scores.
The implementation lives in
[`src/colbert_ppi/scoring.py`](src/colbert_ppi/scoring.py).

Protein–protein and protein–nucleic protocols are separate. The optional
protein–nucleic entry point is:

```bash
colbert-ppi-train-protein-nucleic --config path/to/config.json
```

## Contact-only objective

For each complex, the loader samples labelled contact and non-contact residue
pairs. If the first \(N_+\) entries are matched positives, the sole optimized
objective is:

\[
\mathcal{L}_{contact} =
\frac{1}{2}\left[
\operatorname{CE}(S_{1:N_+,:}, y)
+
\operatorname{CE}(S^\top_{1:N_+,:}, y)
\right].
\]

No protein-level or single-vector objective contributes to the gradient.

## Reproducibility and claim boundary

- Select checkpoints and scoring parameters on validation data only.
- Evaluate the test split once after all choices are frozen.
- Keep protein–protein and protein–nucleic protocols separate.
- Treat late interaction as a system capability in this release; this code
  alone does not establish a causal ranking advantage over single-vector
  baselines.

## Development checks

```bash
python -m compileall -q src tests
pytest -q
ruff check src tests
```

The repository intentionally excludes datasets, checkpoints, third-party
weights, logs, machine addresses, user-specific paths, manuscripts and reviewer
correspondence. Report security or privacy concerns using
[SECURITY.md](SECURITY.md).

## Citation and licence

Citation metadata are provided in [CITATION.cff](CITATION.cff). See
[LICENSE](LICENSE) for the current software-use terms.
