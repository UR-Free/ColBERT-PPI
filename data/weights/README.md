# Model files

Run `python src/download_assets.py ppi` from the repository root.
Optional PRI components use `python src/download_assets.py pri`.
The downloader creates `ppi/` and `pri/` beneath this directory.

Download SaProt separately into `backbones/SaProt_650M_PDB/`.
PRI additionally needs ERNIE-RNA source and weights; configure their paths in
`config/PRI_inference.env`. See [download guide](../../README.md#model-and-benchmark-downloads).
