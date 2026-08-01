# PPI validation/test evidence protocol v3.1

PINDER supplies experimentally resolved positives but not a large
assay-confirmed negative class. Protocol v3.1 follows the database-absence
negative-control convention used by proteome-scale benchmarks such as
RF2-PPI: eligible protein pairs with no prespecified interaction evidence are
used as operational negatives. These controls are not claims of universal
biochemical non-interaction.

RF2-PPI source: Zhang et al., *Science* (2025),
[doi:10.1126/science.adt1630](https://doi.org/10.1126/science.adt1630).

## Frozen labels

- **Positive:** same-organism PINDER structural edge with no train overlap;
  validation edges are also excluded from final test.
- **Operational negative:** eligible same-organism, non-positive pair for which
  both accessions map to STRING and no edge is returned by the frozen network
  query with `required_score=0` (the current API floor was 0.15).
- **Unjudged:** STRING-supported non-PINDER pair or pair with incomplete STRING
  mapping. It remains a rank competitor but is excluded from binary
  AUPRC/AUROC.
- **Strict negative evidence:** exact non-conflicting Negatome 2.0
  manual-stringent pair that is also an operational negative.
- **Ineligible:** cross-organism, missing-taxonomy, train-overlap, or test-time
  validation-overlap cell.

Every cell is recorded in an evidence ledger. The manifest records all input
and output SHA-256 hashes, STRING mapping coverage, query threshold, counts,
and prevalence.

## Metrics

`operational_binary_auprc` and `operational_binary_auroc` compare structural
positives with operational negatives. `auprc` and `observed_label_auprc` are
compatibility aliases of the operational AUPRC. New reporting must use the
explicit operational name and include class counts and prevalence.

`strict_binary_auprc` and `strict_binary_auroc` use only positives plus
Negatome-supported negatives and are `null` when a class is absent. They are a
sensitivity analysis, not a prerequisite for the primary AUPRC.

Primary manuscript values use UniProt max-collapsed entity scores and masks
for operational AUPRC/AUROC as well as rank metrics. Record-level outputs are
ledger/QC results and must be labelled accordingly.

Entity-level bidirectional MRR and Hit@1/5/10/20 are reported after max
collapse of duplicate canonical UniProt records. Checkpoint selection remains
`val_uniprot_max_mrr`; test is evaluated once after all choices are frozen.

The primary fixed-pool AUPRC is not reweighted to RF2-PPI's 1:1000 deployment
prior. Any such prior-adjusted analysis must be separately prespecified.

## Build and evaluate

```bash
colbert-ppi-build-evidence-protocol \
  --train-csv data/processed/train_pairs.csv \
  --val-csv data/processed/val_pairs.csv \
  --test-csv data/processed/test_pairs.csv \
  --uniprot-metadata data/evidence/uniprot_metadata.tsv \
  --string-edges data/evidence/string_edges.tsv \
  --string-mapping data/evidence/string_mapped_accessions.tsv \
  --string-provenance data/evidence/source_provenance.json \
  --negatome data/evidence/negatome2_manual_stringent.txt \
  --output-dir results/ppi_label_protocol_v3_1

colbert-ppi-evaluate-score-matrix \
  --scores path/to/test_scores.npy \
  --pairs data/processed/test_pairs.csv \
  --protocol results/ppi_label_protocol_v3_1/pinder_test_hetero_afdb.evidence_labels.npz \
  --output results/test_metrics.json
```
