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

## 2026-08-01 — Two residue projection heads only

Status: active and binding for the publication-grade model.

The shared SaProt+LoRA backbone feeds exactly two independent per-residue MLP
heads: `query_projector` for query proteins and `candidate_projector` for
candidate partners. The publication path contains no post-SaProt Transformer,
side indicator, random role swap, or residue-weight/attention MLP. Lightweight
epoch checkpoints preserve LoRA, both MLP heads, and the temperature scalar.
