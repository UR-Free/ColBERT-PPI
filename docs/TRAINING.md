# Training, validation, testing and inference

The small-data workflow uses the neural encoders and contact-supervised objective, including trainable SaProt LoRA adapters. It does not replace the encoders with random vectors or train only a final scalar. Each task has 10 training pairs and separate validation and test sets of two pairs each.

## Model files and environment

The training code uses PyTorch, Transformers and PEFT. Package versions are listed in `pyproject.toml` and `requirements-pri.txt`. A CUDA GPU is useful for the neural-network examples; CPU-only code can be used for the cached-vector example and figure-data reproduction.

Download [SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB) from its official distribution. The directory should contain its configuration, tokenizer files and `pytorch_model.bin`. PRI also requires the [official ERNIE-RNA code and pretrained weights](https://github.com/Bruce-ywj/ERNIE-RNA), including its dictionary. The compatible fairseq, Hydra and OmegaConf versions are included in requirements-pri.txt; see ENVIRONMENT.md for the pip compatibility requirement. See `ENVIRONMENT.md` for setup and reference runtimes.

```bash
pip install -r requirements.txt
```

Use paths to your own model files in the commands below. The repository does not include duplicate copies of these public pretrained backbones.

## PPI training

The recommended entry point is `bash PPI_train.sh`; edit `config/PPI_train.env`. The direct Python equivalent is shown below.

```bash
python src/train.py --task ppi \
  --saprot-dir /path/to/SaProt_650M_PDB \
  --data-dir data/examples/training \
  --epochs 1 --batch-size 2 --learning-rate 0.0001 \
  --output runs/ppi
```

The objective samples five positive and five negative residue pairs per complex, applies the symmetric contact InfoNCE loss and the PPI residue-attention regularizer. The contextual encoder has three layers, 16 heads and width 512. SaProt LoRA uses rank 8 and scale 8.

After each epoch, the code builds a reference bank from the mini training set only and selects the checkpoint with the highest validation AUPRC. Ties retain the earliest epoch. Test records are first read after this selection. `best.pt` contains the learned components; `reference_bank.npz` belongs to that same selected checkpoint.

The mini reference bank is necessarily smaller than the paper's 512-identity bank. The demo uses complete short chains, batch size two and one epoch. These settings make the code inspectable with ten training examples; they do not recreate the full training experiment. The paper's settings are retained separately in `config/`.

## PRI training and PPI initialization

Use `bash PRI_train.sh` with `config/PRI_train.env`, or the direct Python command below.

```bash
python src/train.py --task pri \
  --saprot-dir /path/to/SaProt_650M_PDB \
  --ernie-checkpoint /path/to/ERNIE-RNA_pretrain.pt \
  --ernie-code /path/to/ERNIE-RNA \
  --protein-init runs/ppi/best.pt \
  --data-dir data/examples/training --epochs 1 --batch-size 2 \
  --output runs/pri
```

Omit `--protein-init` to train from the pretrained SaProt initialization instead. The transfer loads the PPI contextual encoder, residue-attention head and SaProt LoRA into the protein branch. RNA parameters are not transferred from the PPI model. The PRI contact objective is the same sampled symmetric InfoNCE used by the model, with its training-time attention weights and learned scale. Inference uses unweighted output vectors.

All neural examples exclude terminal special tokens and padding before contextual encoding and pooling. The example training runs write new checkpoints to their output directories.

## Independent evaluation

```bash
python src/evaluate.py --task ppi \
  --saprot-dir /path/to/SaProt_650M_PDB \
  --checkpoint runs/ppi/best.pt \
  --reference-bank runs/ppi/reference_bank.npz \
  --data data/examples/training/ppi_test.json --output runs/ppi_test
```

For PRI, use `--task pri`, its checkpoint and the two ERNIE arguments; no reference bank is needed because α = 0. These tiny retrieval grids treat unobserved cross-pairs as operational negatives for demonstration. They are not the full database-filtered benchmarks used in the paper.

## Inference without contact labels

```bash
python src/predict.py --task ppi \
  --saprot-dir /path/to/SaProt_650M_PDB \
  --checkpoint runs/ppi/best.pt \
  --reference-bank runs/ppi/reference_bank.npz \
  --input data/examples/training/ppi_inference.json \
  --output runs/ppi_predictions.json
```

The inference JSON contains no contact labels. Provide already tokenised inputs in the documented format. For PPI inference from your own structures, see [INPUTS.md](INPUTS.md). Full training-data preprocessing is not included. The full validation and test inputs are supplied in `data/ppi/` and `data/pri/`; the two-pair JSON examples remain small workflow demonstrations.

The ten-example datasets demonstrate the training workflow; their metrics are separate from the full benchmark results.
