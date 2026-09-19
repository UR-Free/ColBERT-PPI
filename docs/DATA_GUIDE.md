# Data guide

## Retrieval

PINDER and Y2H score tables identify both protein endpoints and the evaluation label. AUPRC is calculated as average precision, without trapezoidal interpolation. PINDER negatives are database-filtered operational negatives; Y2H labels come from the experimental screen. Exposure and distance columns define the reported subsets.

Protein–RNA score arrays use unique protein sequences as rows and unique RNA sequences as columns. `proteins.csv` and `rnas.csv` specify their order. Original repeated-record scores were collapsed by the maximum over each pair of exact-entity groups, as defined in Methods. `duplicate_groups.json` records this grouping.

Each cohort has separate positive and negative masks in `labels.npz`. Only their union is evaluated. These masks are identical across models within a cohort, but negative masks need not be nested across cohorts. The common source-excluded cohort contains 576 positives and 57,600 operational negatives. The full cohort contains 612 and 61,200; sequence-distant contains 254 and 25,400; structure-distant contains 138 and 12,881.

`methods.json` maps model/run names to score files. Internal runs comprise nine multi-vector and six single-vector runs. Four additional files contain external comparator scores. Mean performance averages the three runs equally. Relative transfer gain is the ratio of the two means minus one, not the average of per-run ratios.

## Localisation

In each archive, keys start with the original complex index. `a_label` and `b_label` identify interface residues; `a_score` and `b_score` give maximum-over-partner raw cosine scores. Both chains are restricted to experimental construct support. Contact arrays retain all experimentally labelled residue pairs. Chain metrics are averaged within each complex and complexes are weighted equally.

## Source data

Figure and supplementary-table filenames identify their destination. These tables preserve measurements needed to inspect plots, including external comparators and timings. Source tables alone are not a replacement for the original experimental data or full prediction workflow.

## Already processed model inputs

Training accession lists are in `data/training/`. PPI contains 19,386 unique accessions and PRI 1,872. All PRI training records have a recorded UniProt source accession. Full training structures, sequences and contact annotations are not distributed; ten annotated examples per task are supplied separately.

PINDER `validation_inputs.pt` and `test_inputs.pt` are dictionaries keyed by cached protein identity. Each entry contains a one-dimensional `input_ids` tensor including BOS and EOS. The validation and test sets contain 503 and 485 distinct cached inputs, respectively. `*_input_order.csv` explicitly maps the original pair endpoints to those cache keys. This mapping preserves the aliases used in the original evaluation. Inputs retain the evaluation token limit of 1,026 tokens, including special tokens.

PINDER `*_labels.npz` arrays follow the 257-by-257 validation and 248-by-248 test record grids. `receptor_labels` and `ligand_labels` define row and column identities. `positive_mask` and `operational_negative_mask` identify evaluated labels; entries outside the evaluation mask are not automatically negatives. Canonical-entity metrics additionally collapse repeated accessions as specified in Methods and the supplied score tables.

PRI `validation_inputs.pt` and `test_inputs.pt` are lists of 613 and 612 records in their original order. Each record contains `pair_id`, `protein_input_ids`, `nucleic_input_ids`, `positive` and `negative`, with available sequence and source-identity fields. The contact-index tensors refer to residues/nucleotides without special tokens. They are contact supervision, not protein–RNA retrieval labels.

PRI `validation_groups.json` maps original validation records to exact-entity groups. `validation_labels.npz` preserves the complete exact-entity validation grid used for selection. Test retrieval uses the separately frozen cohort masks in `labels.npz` and groups in `duplicate_groups.json`. Do not substitute the validation grid's complement-negative definition for the test cohort masks.

Y2H `test_inputs.pt` contains the prepared token inputs keyed by protein identity. `test_pairs.csv` supplies their experimental pair labels. The files are ready for encoding and do not require the input-preparation scripts.

```python
import torch
import numpy as np

ppi_inputs = torch.load("data/ppi/test_inputs.pt", map_location="cpu", weights_only=True)
ppi_labels = np.load("data/ppi/test_labels.npz")
pri_records = torch.load("data/pri/test_inputs.pt", map_location="cpu", weights_only=True)
```

All token inputs require per-sequence removal of terminal special tokens and padding from residue-level outputs. Supplied input files are for neural encoding; the cached score arrays provide a separate route to check reported metrics without model downloads.
