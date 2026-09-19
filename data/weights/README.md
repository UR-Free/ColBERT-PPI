# Released weights

| Task | Model | Contents |
|---|---|---|
| PPI | ColBERT-PPI, epoch 69 | `ppi/ppi_full/weights.pt`, configuration and required reference bank |
| PRI | 100% PPI initialization, seed 42, epoch 32 | `pri/pri_ppi_100_seed42/weights.pt` and configuration |

Each task archive contains exactly one checkpoint. The PRI model is selected
by the highest final-score validation AUPRC among the three 100% PPI seeds;
see [selection evidence](manifest.json).

Run `python src/download.py ppi` or `python src/download.py pri`
from the repository root. The default inference configurations already point
to these models. Upstream SaProt and, for PRI, ERNIE-RNA are obtained separately.
See the [README](../../README.md#predict-protein-partners).
