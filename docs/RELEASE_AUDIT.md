# Release audit

Release candidate assembled on 2026-07-31.

## Included

- contact-only protein–protein training;
- contact-only protein–nucleic training;
- model and dataset implementations;
- validation/test metric computation;
- canonical bidirectional PPI scoring primitives;
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
- synthetic contact-only tests: 2 passed;
- non-contact implementation scan over Python files: 0 hits;
- absolute private path/host scan: 0 hits;
- personal email scan: 0 hits;
- private-key header scan: 0 hits;
- credential-assignment pattern scan: 0 hits;
- symbolic links: 0;
- files larger than 10 MiB: 0;
- release size before Git metadata: 32 files, 333,526 bytes.

`gitleaks` and `trufflehog` were not installed in the build environment. The
release therefore used explicit regular-expression scans plus an allowlist
review of every tracked path. No data, checkpoints or binary model artifacts
are present.

The uploaded repository was verified as private at
`https://github.com/UR-Free/ColBERT-PPI`; its remote `main` commit matched the
local release commit exactly.
