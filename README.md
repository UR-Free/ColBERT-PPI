# ColBERT-PPI

Structure-aware, residue-explicit protein-pair modelling with SaProt and late
interaction.

[Private repository](https://github.com/UR-Free/ColBERT-PPI) ·
[Data format](docs/DATA_FORMAT.md) ·
[Development decisions](docs/DEVELOPMENT_DECISIONS.md) ·
[Release audit](docs/RELEASE_AUDIT.md)

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

No biological records are bundled. Begin with one manifest per split. Each row
identifies a protein pair and two single-chain PDB files:

```csv
protein_a,protein_b,structure_a,structure_b
complex01_A,complex01_B,structures/complex01_A.pdb,structures/complex01_B.pdb
complex02_A,complex02_B,structures/complex02_A.pdb,structures/complex02_B.pdb
```

Relative PDB paths are resolved from the manifest directory. The two structures
in each row must be extracted from the same complex coordinate frame without
independent rotation or translation; otherwise their cross-chain contact
distances are invalid. Each PDB must contain exactly one recognized protein
chain.

Install Foldseek and prepare a local SaProt checkpoint/tokenizer, then generate
the training artifacts:

```bash
colbert-ppi-prepare-data \
  --manifest manifests/train.csv \
  --output-dir data/processed \
  --prefix train \
  --saprot-dir models/SaProt_650M_PDB \
  --foldseek-bin foldseek \
  --positive-threshold 8 \
  --negative-threshold 12 \
  --max-negatives-per-pair 2048
```

Run the same command for validation and test manifests with `--prefix val` and
`--prefix test`. For each prefix, the command writes:

- `<prefix>_pairs.csv`;
- `<prefix>_saprot_inputs.pt`;
- `<prefix>_cb.npz`;
- `<prefix>_contacts.pt`;
- `<prefix>_summary.json`.

The default contact definition is Cβ distance `<8 Å` for positives and `>12 Å`
for negatives. Glycine uses Cα. Foldseek-derived 3Di and PDB residue sequences
must agree exactly; preparation stops on a mismatch instead of silently
misaligning contact labels.

Copy the example configuration and point it to the generated files:

```bash
cp configs/contact_only_example.json configs/local.json
```

For example, the training fields become:

```json
{
  "pinder_train_csv": "data/processed/train_pairs.csv",
  "pinder_train_saprot_inputs": "data/processed/train_saprot_inputs.pt",
  "pinder_train_cb": "data/processed/train_cb.npz",
  "pinder_train_contact_map": "data/processed/train_contacts.pt"
}
```

Apply the corresponding `val_*` and `test_*` paths in `configs/local.json`.
Keep all generated artifacts outside version control. The serialized schemas
are documented in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

For PINDER-style splits, build the evidence-aware validation/test masks from
the split tables, UniProt organism metadata, STRING association evidence and
Negatome manual-stringent evidence before training. The exact inputs and CLI
are documented in
[docs/PPI_EVALUATION_LABEL_PROTOCOL_V2.md](docs/PPI_EVALUATION_LABEL_PROTOCOL_V2.md).
For a non-PINDER dataset, provide an equivalent order-checked evidence NPZ;
absence from an interaction database must not be converted into a negative.

## Train

After preparing the data and editing `configs/local.json`, start a single-GPU
run:

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

The PPI model uses a shared SaProt+LoRA backbone followed directly by two
per-residue MLP heads: `query_projector` and `candidate_projector`. It contains
no post-SaProt Transformer or residue-weight MLP. When
`save_epoch_components` is enabled, each epoch stores a lightweight LoRA plus
two-head checkpoint under `epoch_components/`.

The example uses a physical batch size of 12 without activation gradient
checkpointing. It does not use micro-batching or gradient accumulation. Adjust
this value only after a full forward/backward memory check on the target GPU.

## Evaluate a frozen checkpoint

```bash
colbert-ppi-evaluate \
  --run-dir outputs/run_YYYYMMDD_HHMMSS \
  --checkpoint best_model.pt \
  --split test \
  --protocol results/ppi_label_protocol_v2/pinder_test_hetero_afdb.evidence_labels.npz \
  --output-dir results/test
```

The evaluator writes `scores.npz` and `metrics.json`. It explicitly evaluates
both role assignments, applies mutual row/column top-`k=1` filtering, sums the
largest `N=10` retained similarities and averages the two orientation scores.
The implementation lives in
[`src/colbert_ppi/scoring.py`](src/colbert_ppi/scoring.py).

PINDER off-diagonal pairs are not assumed to be negatives. The evidence NPZ
separates structural positives, verified negative evidence, unlabelled
candidates, STRING association censoring and ineligible pairs. Primary PINDER
endpoints are UniProt Max-collapsed bidirectional MRR and Hit@1/5/10/20;
positive-versus-unlabelled AUPRC is secondary, and strict binary AUPRC/AUROC is
`null` when either judged class is absent. See
[the evidence-label protocol](docs/PPI_EVALUATION_LABEL_PROTOCOL_V2.md).

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
