# Release history

## 0.2.0 — preprint preparation

The runnable implementation is aligned with the September 2026 manuscript
package. PPI uses the epoch-69 components and their fixed training-reference
bank, with reference calibration (alpha=1, tau=0.03) and smooth MaxSim.
PRI uses alpha=0, tau=0.001. Raw local cosine matrices support localisation.
See [VERSION.md](VERSION.md) for the task-specific settings.

This replaces the earlier contact-training prototype on GitHub. Its mutual
top-k/top-N readout and CLI are not the current manuscript workflow. Earlier
code remains available in Git history; do not mix old configurations or
checkpoints with this version.

The distribution includes a CPU cached-vector example, small annotated
training examples, model inference, benchmark evaluation, optional numerical
reproduction, and preparation of user-supplied single-chain PDBs.
Large data and trained components are separate release assets.
Upstream pretrained backbones come from their original distributors.

The complete original training datasets and end-to-end training preprocessing
are not distributed. The ten-pair demos exercise the implementation;
they do not reproduce the full training experiment.
