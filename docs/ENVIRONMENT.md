# Environment

Linux x86_64 and Python 3.10 are the reference neural environment.
Use an isolated virtual environment and run commands from the repository root.

| Workflow | Installation | External files |
|---|---|---|
| CPU cached-vector example | `pip install -e .` | None |
| PPI inference/training | `pip install -e '.[ppi]'` | SaProt, task components for inference |
| Numerical reproduction | `pip install -e '.[analysis]'` | Benchmark release asset |
| PRI inference/training | `pip install -r requirements-pri.txt` | SaProt, ERNIE-RNA code/weights, task components |

Exact primary dependency versions are in `pyproject.toml`; optional legacy
RNA dependencies are in `requirements-pri.txt`. PPI does not require fairseq.
Install an appropriate PyTorch CUDA build for your driver, or use `DEVICE=cpu`.
The PyTorch dependency can download several GB. No GPU is needed for the CPU
cached-vector demo or frozen-prediction reproduction.

PRI requires Python 3.10 and `python -m pip install 'pip<24.1'` before installing
its requirements: fairseq 0.12.2 depends on legacy OmegaConf metadata rejected
by newer pip. A C/C++ compiler may be needed to build fairseq. For CairoSVG
figure export, install the system Cairo library (e.g. `libcairo2` on Ubuntu).

## Common errors

- Missing `pytorch_model.bin`: download the Hugging Face SaProt model directory,
  not only the upstream standalone `.pt` artifact.
- Missing components/reference bank: run the asset downloader. Keep a model's
  `weights.pt`, `config.json` and `reference_bank.npz` together.
- CUDA unavailable or out of memory: use `DEVICE=cpu`, select a free GPU using
  `CUDA_VISIBLE_DEVICES`, or use shorter explicit input crops. Memory depends
  on sequence lengths; no universal GPU memory minimum is claimed.
- Missing benchmark files: download the `benchmarks` asset before evaluation.
- Release URL returns 404: check that the release has been published. During
  private staging, use authenticated GitHub CLI download, then `--archive`.
- SHA-256 mismatch: use the exact asset matching this checkout's `assets.json`.

Reference local smoke tests used four PyTorch CPU threads and float32; model
loading plus one-pair inference took approximately one minute. These timings
exclude installation/downloads and do not imply a full-benchmark runtime.
Current validation scope is recorded in [VALIDATION.md](VALIDATION.md).
