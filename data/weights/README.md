# Model files

Run `python scripts/download_assets.py ppi` from the repository root.
Optional PRI components use `python scripts/download_assets.py pri`.
The downloader creates `ppi/` and `pri/` beneath this directory.

Download SaProt separately into `backbones/SaProt_650M_PDB/`.
PRI additionally needs ERNIE-RNA source and weights; configure their paths in
`config/PRI_inference.env`. See [download guide](../../docs/WEIGHTS.md).
