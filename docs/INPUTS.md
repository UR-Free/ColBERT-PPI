# Use your own proteins

The complete PPI model uses an amino-acid/3Di token per represented residue.
An amino-acid FASTA alone does not supply the structural input required by
this model. Use experimental or predicted individual protein structures.

1. Download the official [SaProt_650M_PDB backbone](https://huggingface.co/westlake-repl/SaProt_650M_PDB),
   including `config.json`, `pytorch_model.bin`, `vocab.txt`,
   `tokenizer_config.json` and `special_tokens_map.json`.
2. Install [Foldseek](https://github.com/steineggerlab/foldseek#installation).
   The preparation command uses `structureto3didescriptor`.
3. Extract one protein chain per PDB file. Avoid multiple models or multiple
   chains in one file. For inference, the two monomers do not need a common
   coordinate frame or a known complex structure.
4. Create a CSV. Paths are relative to the CSV directory (absolute paths also work):

```csv
pair_id,structure_a,structure_b
my_pair,structures/protein_a.pdb,structures/protein_b.pdb
```

```bash
python src/prepare_pairs.py --manifest pairs.csv \
  --saprot-dir data/weights/backbones/SaProt_650M_PDB \
  --foldseek-bin foldseek --output data/user/pairs.json
INPUT=data/user/pairs.json OUTPUT=data/predictions/my_pairs.json bash PPI_inference.sh
```

The output JSON contains `pair_id`, `left_tokens` and `right_tokens`. Token
arrays include BOS=0 and EOS=2; padding=1 is added during batching. Preparation
fails on ambiguous chains, unknown tokens, or more than 1,024 residues.
It does not silently truncate. Crop or split long proteins deliberately and
retain a mapping to original residue numbering. Low-confidence residues are
not automatically masked; use a consistent structure/preprocessing policy
when comparing candidates.

The prediction file is a list of `{ "pair_id": "my_pair", "score": ... }`
records. Higher scores rank candidates more strongly; values are not
probabilities and no universal interaction threshold is supplied. The same
checkpoint and reference bank must be used across the candidate comparison.

## Protein–RNA inputs

PRI accepts the same JSON keys, with SaProt IDs on the left and ERNIE-RNA
IDs on the right. Use the vocabulary and token conventions of the specified
upstream ERNIE-RNA checkpoint; do not use SaProt or generic DNA tokenizer IDs
for RNA. A ready-to-run tokenized example is in
`data/examples/training/pri_inference.json`. Automated preparation of new raw
RNA sequences is not included in this release.

## Training labels

Training examples additionally contain `positive` and `negative` lists of
zero-based residue-index pairs, excluding BOS/EOS. These are contact labels,
not protein-level interaction labels. Contact supervision requires structures
in the same experimental complex frame; independently placed predicted
monomers cannot establish contact labels. See [training](TRAINING.md).
