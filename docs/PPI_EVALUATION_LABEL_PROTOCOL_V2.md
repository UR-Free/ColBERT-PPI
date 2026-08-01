# PPI validation/test evidence protocol v2

> Historical protocol. Use `PPI_EVALUATION_LABEL_PROTOCOL_V3.md` (v3.1) for the
> reportable database-absence operational-negative AUPRC.

## Why this protocol exists

The PINDER validation and test files contain experimentally resolved positive
complexes. Their off-diagonal Cartesian products have not been tested for
non-interaction. Treating every off-diagonal cell as a true negative therefore
creates a closed-world label error and turns PINDER retrieval into a
positive--unlabelled (PU) problem.

Protocol v2 does not infer non-interaction from database absence. It stores
positive, verified-negative, unlabelled, eligible-candidate, and censored masks
separately and preserves their evidence provenance.

## Label adjudication

### Structural positive

A pair is a primary positive only when all of the following hold:

1. the UniProt pair occurs as an experimentally resolved PINDER complex in the
   relevant split;
2. duplicate records carrying the same unordered canonical UniProt pair are
   expanded to the same positive entity edge;
3. both proteins belong to the same normalized UniProt organism group;
4. the edge does not overlap the training set; and
5. for the final test, the edge does not overlap validation.

Cross-species PINDER complexes remain in the evidence ledger but are excluded
from the primary within-species retrieval analysis. They require a dedicated
host--pathogen candidate universe and cannot be mixed with ordinary
within-proteome decoys.

The organism-group key is deterministic: whitespace and case are normalized,
and UniProt parenthetical strain/isolate annotations are removed. The exact
rule is recorded in the manifest. This avoids silently mixing different
organisms while grouping strain-qualified records under the same organism
name; it is not presented as an NCBI taxonomy-rank inference.

### Verified negative evidence

An exact canonical UniProt pair can be labelled negative only when it occurs in
the Negatome 2.0 `manual-stringent` file and has no conflicting structural,
STRING-association, training-edge, or validation-edge evidence. These negatives
remain assay- and context-dependent; they are not claims that the proteins can
never interact under any biological condition.

In the current AFDB/ESM-Atlas-filtered PINDER splits, this strict rule yields no
eligible verified negatives. That result is retained rather than replacing the
missing negatives with random pairs.

### Unlabelled candidate

All remaining same-species, non-leaking candidates are unlabelled. They remain
competitors for rank-based retrieval metrics but must not be described as
confirmed non-interactions.

### STRING association censor

STRING edges with combined score at least 0.7 are treated only as evidence that
an off-diagonal cell may represent a known biological association. Such cells
are censored from positive-vs-unlabelled AUPRC. STRING does not promote a pair
to a direct physical-interaction positive.

## Primary and secondary endpoints

The primary PINDER endpoints are entity-level, bidirectional rank metrics:

- mean reciprocal rank of the first known structural partner;
- Hit/Recall@1, @5, @10, and @20 (with both query directions also retained);
- number of evaluable queries and eligible candidates.

The candidate universe is restricted to same-species proteins. Queries with
fewer than two candidates are excluded as trivial retrieval problems. Duplicate
UniProt records are max-collapsed before the primary metric is reported.
Legacy Hit@100/200/300 fields remain in machine outputs for compatibility but
are not primary endpoints because they saturate these candidate pools.

`observed_label_auprc` is retained only as a secondary PU sensitivity. It means
known structural positives versus unlabelled candidates after censoring known
associations. It is not a true-negative binary AUPRC.

`strict_binary_auprc` and `strict_binary_auroc` are computed only over structural
positives plus verified Negatome negatives. They are JSON `null` when either
class is absent. The current PINDER val/test protocol therefore does not report
a strict binary AUPRC.

The independent Lambourne et al. Y2H benchmark remains the binary-assay test:
its negative class means assay non-detection, not universal biochemical
non-interaction. PINDER and Y2H answer different questions and are reported
separately.

## Checkpoint and test firewall

Future matched ColBERT-PPI and FlashPPI runs use
`val_uniprot_max_mrr` for checkpoint selection. Test data are not loaded during
training and are evaluated once after the validation-selected checkpoint is
frozen. Contact/interface labels and all test metrics are excluded from model
selection.

## Rebuild

From the repository root:

```bash
colbert-ppi-build-evidence-protocol \
  --train-csv data/processed/pinder_train.csv \
  --val-csv data/processed/pinder_val.csv \
  --test-csv data/processed/pinder_test.csv \
  --uniprot-metadata data/evidence/uniprot_metadata.tsv \
  --string-edges data/evidence/string_edges.tsv \
  --negatome data/evidence/negatome2_manual_stringent.txt
```

Default outputs:

- `results/ppi_label_protocol_v2/protocol_manifest.json`
- `results/ppi_label_protocol_v2/pinder_val_hetero_afdb.evidence_labels.npz`
- `results/ppi_label_protocol_v2/pinder_test_hetero_afdb.evidence_labels.npz`
- one evidence ledger CSV per split

The manifest records SHA-256 hashes for every input and the exact label counts.
Training configs opt in using `val_label_protocol` and
`test_label_protocol`; these fields supersede the legacy
`*_calibrated_pairs` CSVs.

To evaluate a ColBERT-PPI, FlashPPI, or other baseline score matrix under the
identical test protocol:

```bash
colbert-ppi-evaluate-score-matrix \
  --scores path/to/pinder_test_scores.npy \
  --pairs data/processed_pinder_afdb_esmatlas/pinder_test_hetero_afdb.csv \
  --protocol results/ppi_label_protocol_v2/pinder_test_hetero_afdb.evidence_labels.npz \
  --output path/to/pinder_test_evidence_metrics.json
```

The command aborts if matrix shape, pair order, labels, or finite-score checks
fail, and records SHA-256 hashes for the scores, pair table, and protocol.
