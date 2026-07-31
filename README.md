# ColBERT-PPI

[Repository](https://github.com/UR-Free/ColBERT-PPI)

ColBERT-PPI is a structure-aware, residue-explicit framework for protein-pair
scoring. Each protein is encoded independently from SaProt amino-acid/3Di
tokens; contextualized residue embeddings are compared by late interaction to
produce a protein-pair score and pair-conditioned regional evidence.

## Scope

This repository contains publication-grade model, training and evaluation
code. It deliberately excludes:

- raw or processed biological datasets;
- model checkpoints and third-party model weights;
- training logs, machine addresses and user-specific paths;
- manuscript drafts and unpublished reviewer correspondence.

The repository is private while the associated manuscript and archival release
are prepared.

## Contact-only optimization objective

The sole optimization objective is sampled bidirectional residue-contact
InfoNCE. For each complex, the loader samples up to five labelled contact pairs
and five labelled non-contact pairs. The first \(N_+\) entries are matched
positive residue pairs:

\[
\mathcal{L}_{contact} =
\frac{1}{2}\left[
\operatorname{CE}(S_{1:N_+,:}, y)
+
\operatorname{CE}(S^\top_{1:N_+,:}, y)
\right].
\]

There is no protein-level InfoNCE, auxiliary ranking surrogate, attention
regularizer, self-hard-negative objective or pooled single-vector path. AUPRC,
AUROC, MRR and Hit@K are evaluation metrics only.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

SaProt weights are not redistributed. Download an upstream SaProt checkpoint
under its original license and set `saprot_dir` in the experiment config.

## Data preparation

No biological records are bundled. A training example requires:

1. a CSV containing paired protein identifiers;
2. pre-tokenized SaProt inputs keyed by protein identifier;
3. Cβ coordinates keyed by protein identifier;
4. sparse contact labels keyed by pair identifier.

The exact schemas are documented in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Training

Copy and edit the example configuration, then run:

```bash
python scripts/train.py \
  --config configs/contact_only_example.json \
  --gpu 0
```

Training always uses contact InfoNCE. Evaluation always uses residue-level late
interaction.

## Evaluation

The frozen publication PPI scorer is implemented in
`scripts/analysis/canonical_scoring.py`: it evaluates both role assignments
explicitly, applies mutual-top-1/top-10 within each assignment, and averages
the two orientation scores.

## Reproducibility and claim boundary

- Checkpoints are selected using validation data only.
- Test data must not be used for checkpoint or operating-point selection.
- Protein–protein and protein–nucleic evaluation protocols are separate.
- The code supports a system-level joint capability claim; it does not by
  itself identify a causal accuracy advantage of late interaction.

See [docs/DEVELOPMENT_DECISIONS.md](docs/DEVELOPMENT_DECISIONS.md) for binding
implementation decisions and [docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md) for
the release audit.
