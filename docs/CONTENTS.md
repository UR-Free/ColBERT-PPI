# Data and model contents

| Location | Contents |
|---|---|
| `data/examples/training/` | Ten annotated training pairs per task and separate two-pair validation/test examples |
| `data/training/` | Training protein UniProt identifiers |
| `data/ppi/`, `data/pri/`, `data/y2h/` | Processed benchmark inputs, labels and entity definitions |
| `data/source_data/` | Numerical figure and table data |
| `data/localisation/` | Local scores and experimental contact labels |
| `data/figure_templates/` | Editable final figure compositions |
| `data/validation_reports/` | Checkpoint-selection tables and numerical reproduction outputs |
| `data/weights/` | Location for separately downloaded model components and backbones |

The model archives contain three PPI checkpoints and fifteen PRI checkpoints. Multi-vector PPI checkpoints include their corresponding training-reference banks. The upstream SaProt and ERNIE-RNA backbones are downloaded separately.

The complete annotated training datasets and training-data preprocessing scripts are not included. UniProt identifiers alone do not specify training pairs, contact labels or crops; the small training examples therefore do not reproduce the full training experiment. Benchmark inputs are already processed and can be evaluated directly.

Use the entry points in `src/` to run the packaged implementation. Earlier research implementations remain in Git history.

Large benchmark folders are installed by `python scripts/download_assets.py benchmarks`; model components are separate `ppi` and `pri` assets.
