# 🧬 ColBERT-PPI

Retrieve protein partners with reusable residue-level representations.
Supports protein–protein (PPI) and protein–RNA (PRI) matching.

![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![CUDA GPU](https://img.shields.io/badge/CUDA-GPU-76B900?logo=nvidia&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[![bioRxiv — Read the preprint](data/paper/biorxiv-badge.svg)](https://www.biorxiv.org/content/10.64898/2026.09.19.752878v1)

[Quick start](#-quick-start) · [Evaluate](#-evaluate) ·
[Model weights](https://github.com/UR-Free/ColBERT-PPI/releases/tag/v0.2.1-preprint) · [For agent use](#-for-agent-use)

[![Fig. 1 — ColBERT-PPI](data/paper/Fig1.png)](data/paper/Fig1.svg)

*Contact-supervised representations for partner retrieval and interface evidence.
Click the figure for the vector version.*

## 🚀 Quick start

**Requirements:** Linux · Conda · CUDA GPU

```bash
# Get the code
git clone https://github.com/UR-Free/ColBERT-PPI.git
cd ColBERT-PPI

# Create the environment and install dependencies
conda create -n colbert-ppi python=3.10 -y
conda activate colbert-ppi
pip install -r requirements.txt

# Download PPI model weights and the reference bank into data/weights/ppi/
python src/download.py ppi
```

Download [SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB)
into `data/weights/backbones/SaProt_650M_PDB/`, including its configuration,
weights and tokenizer files. Run the bundled protein-pair example:

```bash
# Score the bundled protein pairs on GPU using config/ppi.json
python src/predict.py
```

> **GPU:** Defaults to `cuda:0`. Set `CUDA_VISIBLE_DEVICES=1` to select another GPU.

**Output:** `data/results/ppi/predictions.json`. Higher scores indicate stronger
matches; scores are not probabilities.

For your own proteins, follow the [PDB input example](data/README.md#custom-ppi-inputs), then:

```bash
# Score your prepared protein pairs instead of the bundled example
python src/predict.py --input data/user/pairs.json
```

## 📊 Evaluate

```bash
# Download benchmark inputs, labels and saved predictions into data/benchmarks/
python src/download.py benchmarks

# Run the PPI model on the test set and calculate retrieval metrics
python src/evaluate.py --task ppi --split test
```

This runs the model on GPU and saves results under `data/results/ppi/evaluation/`.
To recalculate paper metrics from the supplied predictions instead:

```bash
# Recalculate metrics from saved predictions, without running the model
python src/reproduce.py
```

Only two checkpoints are distributed: **ColBERT-PPI epoch 69** and **PRI with
100% PPI initialization, seed 42, epoch 32**. The PRI checkpoint has the best
validation AUPRC among the three 100% PPI seeds. Downloads verify the
[asset checksums and model identities](data/weights/manifest.json).

<details>
<summary>🛠️ <strong>Protein–RNA, training and manual downloads</strong></summary>

PRI additionally uses the official [ERNIE-RNA source and weights](https://github.com/Bruce-ywj/ERNIE-RNA).
Set their paths in `config/pri.json`, then:

```bash
# Use a pip version compatible with the older fairseq dependencies
pip install 'pip<24.1'
pip install -r config/requirements-pri.txt

# Download the selected protein–RNA model weights into data/weights/pri/
python src/download.py pri

# Score the bundled protein–RNA pairs on GPU using config/pri.json
python src/predict.py --task pri
```

The older fairseq dependency requires Python 3.10 and may need a C/C++ compiler.
PRI examples are tokenized; raw RNA preprocessing is not included.

`python src/train.py --task ppi` (or `pri`) runs a one-epoch demo on ten pairs.
Full original training data and its preprocessing are not included; demo
metrics are not manuscript benchmark results.

For an existing release ZIP, use `python src/download.py ppi --archive /path/to/file.zip`.
Extraction refuses to overwrite existing files. While the repository is private,
downloading requires an authorized GitHub account (`gh auth login`) or a manually
downloaded archive.

</details>

---

[Data formats](data/README.md) · [MIT license](LICENSE) ·
[Citation](data/paper/CITATION.cff) · [Issues](https://github.com/UR-Free/ColBERT-PPI/issues)

## 🤖 For agent use

- Work from the repository root in the `colbert-ppi` Conda environment.
  Install `requirements.txt`; PRI also needs `config/requirements-pri.txt`.
- Check CUDA availability and select a free GPU. Model workflows default to
  GPU; report CUDA problems rather than silently switching to CPU.
- Fetch assets with `python src/download.py`; keep checkpoints and reference
  banks matched to `data/weights/manifest.json`. Configure backbone paths in `config/`.
- Run `python src/check_example.py` for a lightweight scoring check (expect `PASS`),
  then `python src/predict.py` to verify actual GPU inference. Do not start training
  as an installation check. Development checks: `pip install pytest` and `pytest -q -c src/pytest.ini`.
- Distinguish `src/reproduce.py` (saved predictions) from `src/evaluate.py` (model inference).
  Keep test masks and scoring fixed; do not tune on test data or treat unjudged pairs as negatives.
- Run lengthy jobs detached. Report the command, environment, GPU, model,
  output paths and checks actually completed. Generated files belong in `data/results/`.
