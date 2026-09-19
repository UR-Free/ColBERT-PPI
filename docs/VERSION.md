# Model and scoring configuration

| Component | Configuration |
|---|---|
| PPI complete model | Epoch 69 |
| PPI sequence-only model | Epoch 50 |
| PPI single-vector model | Epoch 74 |
| Multi-vector PPI and Y2H | Reference calibration, α = 1, τ = 0.03 |
| Multi-vector PRI | Smooth MaxSim, α = 0, τ = 0.001 |
| Interface localisation | Raw residue cosine matrix, averaged over encoder roles |
| PRI input masks | Per-sequence exclusion of terminal special tokens and padding |

The weight archives' `models.csv` records the selected epoch for each PRI run. Full model settings are stored alongside the weights; launch defaults are in `config/`.
