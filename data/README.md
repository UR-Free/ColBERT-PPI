# Data and model files

| Directory | Contents | Acquisition |
|---|---|---|
| `examples/` | Frozen-vector CPU example and expected matrices | Included |
| `examples/training/` | 10 training and separate 2-pair validation/test examples per task | Included |
| `training/` | Source protein accession lists | Included |
| `weights/ppi/`, `weights/pri/` | Selected learned components and model indexes | `python src/download_assets.py ppi` / `pri` |
| `ppi/`, `pri/`, `y2h/` | Prepared full benchmark tokens, labels and entity mappings | `benchmarks` asset |
| `source_data/`, `localisation/` | Frozen scores, contact labels and numerical figure evidence | `benchmarks` asset |
| `figure_templates/` | Final editable SVG compositions | `benchmarks` asset |
| `validation_reports/` | Checkpoint-selection and numerical reproduction outputs | `benchmarks` asset |

Download commands run from the repository root. Frozen backbone weights are
not included. Each learned `weights.pt` stores `model`, `epoch` and `model_id`;
keep it with its `config.json` and reference bank, where applicable.

## Input and label conventions

Inference JSON fields are `pair_id`, `left_tokens`, `right_tokens`; tokens
include BOS=0 and EOS=2. Training JSON also has `positive` and `negative`
zero-based residue-pair indices excluding special tokens. Contact negatives
are distinct from protein-level retrieval negatives. Mini training splits are
workflow examples, not the manuscript benchmark cohorts.

PINDER `*_inputs.pt` dictionaries contain `input_ids` including terminal tokens.
`*_input_order.csv` maps pair endpoints to cache keys. Label NPZ arrays follow
257-by-257 validation and 248-by-248 test record grids; accession duplicates
are collapsed by maximum scores and logical-OR positive labels. Operational
negatives are database-filtered, not necessarily experimentally established
non-interactions. Unjudged pairs are excluded rather than relabelled negative.

PRI records have `protein_input_ids` and `nucleic_input_ids`. Exact-entity
sequence groups define the retrieval grid. Duplicate scores use the maximum;
`labels.npz` supplies the frozen cohort masks. Only the union of positive and
negative masks is evaluated. The common source-excluded cohort contains 576
positive and 57,600 operational-negative pairs. Validation complement labels
must not replace the test masks. Y2H labels come from its experimental screen.

Average precision is used for AUPRC without trapezoidal interpolation.
Localisation arrays preserve experimentally supported chain/interface labels;
complexes are weighted equally. The bundled reproduction script checks saved
predictions; it does not train encoders, rebuild original datasets or remeasure
hardware timings. Accession lists alone cannot reconstruct training pairs,
contact labels or crops. Full original training-data preparation is not released.

## Upstream sources and terms

- [SaProt](https://github.com/westlake-repl/SaProt) and
  [SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB): protein backbone and vocabulary.
- [ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA): optional RNA backbone, dictionary and code.
- [Foldseek](https://github.com/steineggerlab/foldseek): structure-to-3Di preprocessing.
- [PINDER](https://github.com/pinder-org/pinder): source of processed PPI examples and benchmark inputs.
- [UniProt](https://www.uniprot.org/): accession identifiers.

MIT terms for author-created code and learned components do not relicense
underlying third-party data, pretrained models or dependencies. Obtain
backbones from their original distributors and follow their supplied terms.
Comparator predictions are frozen evidence, not redistributed comparator
software or model weights. Preserve source identifiers and refer to the
manuscript and source tables for experimental provenance.

## Custom PPI inputs

Install [Foldseek](https://github.com/steineggerlab/foldseek#installation) and
prepare one protein chain per PDB file. Create `pairs.csv`:

```csv
pair_id,structure_a,structure_b
my_pair,structures/protein_a.pdb,structures/protein_b.pdb
```

Paths are relative to the CSV. For inference, the monomers need not share a
coordinate frame. Amino-acid FASTA alone is not sufficient for the full
structure-aware model.

```bash
python src/prepare_pairs.py --manifest pairs.csv \
  --saprot-dir data/weights/backbones/SaProt_650M_PDB \
  --foldseek-bin foldseek --output data/user/pairs.json
INPUT=data/user/pairs.json OUTPUT=data/predictions/my_pairs.json \
  bash src/launchers/PPI_inference.sh
```

Unknown tokens, ambiguous chains or more than 1,024 represented residues are
errors. Crop long proteins explicitly and retain a mapping to original
numbering; no silent truncation occurs. Structure quality and missing residues
can affect scores. Output scores are relative matching evidence, not binding
probabilities or affinities, and no universal binary threshold is supplied.

## Released readouts

The September manuscript implementation supersedes the historical GitHub
top-k/top-N prototype. PPI uses epoch 69 with its fixed reference bank,
alpha=1 and tau=0.03; PRI uses alpha=0 and tau=0.001. Both PPI encoder-role
assignments are evaluated explicitly before score averaging. Localisation
uses raw residue cosine matrices, not calibrated retrieval matrices. It does
not establish biochemical causality. Keep checkpoints, banks, preprocessing
and readout settings matched when comparing candidates.
