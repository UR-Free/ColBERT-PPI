# Model and data downloads

Assets for this code version belong to
[v0.2.0-preprint](https://github.com/UR-Free/ColBERT-PPI/releases/tag/v0.2.0-preprint).
SHA-256 checksums and immutable versioned URLs are recorded in `assets.json`.

```bash
python scripts/download_assets.py ppi
python scripts/download_assets.py pri          # optional RNA models
python scripts/download_assets.py benchmarks   # optional evaluation data
```

PPI includes three selected models: complete (epoch 69), sequence-only (epoch
50), and single-vector (epoch 74), with reference banks for multi-vector
models. PRI includes fifteen models and their `models.csv` index.
Benchmark data includes prepared inputs, labels, frozen predictions,
localisation evidence, source tables and final figure templates.

The archives extract from the repository root into `data/weights/` or `data/`.
The downloader verifies the entire archive before extraction and refuses to
overwrite existing files. To inspect a second copy, use `--destination`.
For manual downloads, obtain the ZIP from the release page and run:

```bash
python scripts/download_assets.py ppi --archive /path/to/ColBERT-PPI-ppi-v0.2.0.zip
```

Private staging requires authenticated download, for example
`gh release download v0.2.0-preprint --repo UR-Free/ColBERT-PPI --pattern 'ColBERT-PPI-ppi-v0.2.0.zip'`.
Published public assets require no account.

## External backbones

Obtain [SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB)
separately. PRI also needs the [ERNIE-RNA pretrained model and source](https://github.com/Bruce-ywj/ERNIE-RNA).
Follow their upstream terms. Frozen backbone weights are not redistributed.

Each `weights.pt` stores learned tensors under `model`, plus `epoch` and
`model_id`. Optimizer states are omitted. Missing frozen-backbone keys during
component loading are expected; missing learned parameters are errors.

Use `src/predict.py` for complete multi-vector pair inference. Use
`src/benchmark.py` for the supplied sequence-only and single-vector controls.
Keep `config.json` next to the checkpoint; benchmark mode reads the
architecture and task settings there. See [VERSION.md](VERSION.md).
