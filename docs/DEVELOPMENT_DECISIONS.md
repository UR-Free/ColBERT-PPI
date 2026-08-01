# Persistent development decisions

## 2026-07-31 — Contact-only optimization objective

Status: active and binding for all publication-grade code.

The only permitted training loss is the sampled bidirectional residue-contact
InfoNCE objective (`contact_loss`).

The following training objectives and regularizers were removed:

- protein-level InfoNCE;
- residue-pair AUROC surrogate loss;
- interface-residue AUROC surrogate loss;
- attention sparsity/binarization regularization;
- self-hard-negative loss augmentation.

Evaluation metrics such as AUPRC, AUROC, MRR and Hit@K remain permitted because
they are read-only measurements, not optimization objectives. Single-vector
pooling and scoring are not part of the publication-grade code.

Any future change that introduces an additional optimization term must be
documented here before it enters publication-grade code.

## 2026-08-01 — Database-absence operational negatives for PINDER

Status: active and binding for confirmatory validation and test evaluation.

Structural PINDER edges define positives. Eligible same-organism pairs absent
from both that structural-positive set and the frozen STRING association set
define operational negative controls. This enables a reportable
`operational_binary_auprc`, but the controls must not be described as proven
biochemical non-interactions. Exact non-conflicting Negatome manual-stringent
edges remain a stricter sensitivity tier. STRING-supported non-PINDER pairs are
censored from binary metrics but remain rank competitors. Every result reports
the operational pool's counts and prevalence plus UniProt Max-collapsed MRR and
Hit@1/5/10/20. Checkpoints are selected on validation only; test is evaluated
after all choices are frozen.

## 2026-08-01 — Two residue projection heads only

Status: active and binding for the publication-grade model.

The shared SaProt+LoRA backbone feeds exactly two independent per-residue MLP
heads: `query_projector` for query proteins and `candidate_projector` for
candidate partners. The publication path contains no post-SaProt Transformer,
side indicator, random role swap, or residue-weight/attention MLP. Lightweight
epoch checkpoints preserve LoRA, both MLP heads, and the temperature scalar.
