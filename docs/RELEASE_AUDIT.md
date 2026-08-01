# Release audit

Release candidate assembled on 2026-07-31.

## Included

- contact-only protein–protein training;
- contact-only protein–nucleic training;
- model and dataset implementations;
- validation/test metric computation;
- canonical bidirectional PPI scoring primitives;
- installable `colbert_ppi` package and four user-facing CLI entry points;
- synthetic unit tests and an example configuration.

## Excluded

- `data/`, `outputs/`, `results/`, `figures/` and manuscript files;
- checkpoints and pretrained third-party weights;
- logs, cache files and environment snapshots;
- absolute local paths, hostnames, credentials and personal email addresses;
- historical training objectives and their experiment configurations.

## Verification

Verified on 2026-07-31:

- Python compile check: passed;
- contact-only and data-preparation unit tests: 4 passed;
- Ruff source/test check: passed;
- wheel build and package discovery: passed;
- CLI help smoke checks: 5 passed;
- non-contact implementation scan over Python files: 0 hits;
- absolute private path/host scan: 0 hits;
- personal email scan: 0 hits;
- private-key header scan: 0 hits;
- credential-assignment pattern scan: 0 hits;
- symbolic links: 0;
- files larger than 10 MiB: 0;
- release size before Git metadata: 33 files, 352,818 bytes.

`gitleaks` and `trufflehog` were not installed in the build environment. The
release therefore used explicit regular-expression scans plus an allowlist
review of every tracked path. No data, checkpoints or binary model artifacts
are present.

The uploaded repository was verified as private at
`https://github.com/UR-Free/ColBERT-PPI`; its remote `main` tree was verified
byte-for-byte against the local release tree. Commit object IDs may differ
because reviewed files are uploaded atomically through the GitHub API.

## 2026-08-01 evidence-protocol update

- added three-state PINDER labels, full-cell evidence ledgers and hashed
  protocol manifests;
- added model-agnostic score-matrix evaluation for matched ColBERT-PPI,
  FlashPPI and other baselines;
- changed confirmatory checkpoint selection to entity-level validation MRR;
- Python compile check passed;
- 11 tests passed;
- CLI help smoke checks passed for the two new entry points;
- whitespace and private-path/host/credential scans passed.

Ruff was not installed in the active SaProt environment for this update; the
earlier release-level Ruff result remains historical rather than being claimed
for the new diff.

## 2026-08-01 operational-negative v3 update

- changed the reportable binary endpoint to structural positives versus
  prespecified database-absence operational negative controls;
- retained Negatome-only strict metrics as a sensitivity tier;
- added explicit AUPRC/AUROC, negative-count and prevalence outputs;
- preserved validation-MRR checkpoint selection and the final-test firewall;
- synchronized the model-agnostic evaluator, builder, documentation and tests.
- remote private-repository commit after this update:
  `8160eb1aa8c623d7d938af61f4d65b2a0922482d`; its tree matched the reviewed
  local tree `bb75e60591b5cde97361fa655bc312a642807c24` before this audit-note-only
  follow-up.
- v3.1 then tightened absence to a frozen STRING `required_score=0` query and
  requires successful mapping of both accessions; unmapped pairs are unjudged.
