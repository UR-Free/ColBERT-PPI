# Release validation

Local verification on 2026-09-19, Linux x86_64:

| Check | Result |
|---|---|
| Fresh Python 3.10 virtual environment, `pip install -e '.[dev]'` | PASS |
| Core tests (scoring invariants, masks, invalid input, PDB preparation, safe asset extraction) | 21 passed |
| Cached-vector example, including frozen matrix comparison | PASS |
| Wheel and source distribution build | PASS |
| Release PPI ZIP checksum and extraction | PASS |
| Python compilation and four shell syntax/dry-run checks | PASS |
| Frozen-prediction metric reproduction | PASS; 226 localisation checks per model, 76 PRI run/cohort combinations |
| PPI neural inference on the two bundled pairs | PASS |
| PRI neural inference on the two bundled pairs | PASS |
| Single-chain PDB → Foldseek → SaProt tokens → PPI prediction | PASS; 2PTC chains E/I |

Exact smoke predictions and scope are in [validation.json](validation.json).
The PPI example scores were approximately 0.26619631 and 0.24707108; PRI
scores were 0.36796659 and 0.00440597. These are smoke outputs, not performance
estimates. Small floating-point differences across devices are expected.

Only the lightweight core was installed into a fresh environment. Neural
checks reused the existing research environment with PyTorch 2.8.0,
Transformers 5.12.1 and PEFT 0.19.1, using CPU float32 and four threads.
No full training, full neural benchmark, clean neural installation, GPU test,
or independent hardware timing replication was performed for this release.
The CPU PRI check emitted optional tensorboardX/NVML notices but completed.

A ready-to-use GitHub Actions template is in `docs/ci/cpu.yml` for Python
3.10 and 3.12. It is not active: the publishing credential lacks the GitHub
`workflow` scope. A maintainer can copy it to `.github/workflows/cpu.yml`
using the GitHub editor or a credential with that scope. No CI run is claimed.
