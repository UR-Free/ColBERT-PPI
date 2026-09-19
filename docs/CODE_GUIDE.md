# Reading the code

Start with the four shell entry points in the repository root. Each reads its matching `config/*.env` file and calls a Python entry point in `src/`. Then follow either the inference path or the training path. Neural model definitions are separate from data loading and experiment entry points.

## Scoring and reported results

1. `src/colbert_ppi/scoring.py` defines reference correction and smooth MaxSim. The PPI function returns both the retrieval score and the raw localisation matrix.
2. `src/run_example.py` applies the rule to one saved protein-pair example.
3. `src/reproduce_results.py` calls three checks from `src/colbert_ppi/evaluation.py`: retrieval, localisation and protein–RNA transfer. It writes recalculated metrics to the chosen output directory.

The metric checks consume saved predictions. They do not construct a model or run training.

## Neural inference

`model_loading.py` constructs the architecture and restores its learned components. `inference.py` encodes already prepared token inputs. `scoring.py` compares the resulting vectors. `src/predict.py` connects these steps for label-free prediction; `src/evaluate.py` adds the demonstration retrieval labels and metrics.

| Module | Responsibility |
|---|---|
| `neural/ppi.py` | SaProt, LoRA and the shared role-conditioned protein encoder |
| `neural/pri.py` | Protein branch, RNA branch and residue–nucleotide outputs |
| `neural/ppi_context.py`, `neural/pri_context.py` | Contextual attention layers for the corresponding architectures |
| `neural/rna_backbone.py`, `neural/lora.py` | RNA backbone integration and adapter support |
| `model_loading.py` | Model construction, component loading and protein-branch transfer |
| `inference.py` | Encoding, the small reference bank and demo retrieval evaluation |
| `training_data.py` | Reading prepared JSON examples and batching their contact labels |
| `neural/losses.py` | Contact-supervised training objective |
| `training.py` | Optimization, validation checkpoint selection and held-out evaluation |

The two context modules preserve the respective trained architectures; they are not interchangeable merely because their class names resemble each other.

## Training demo

Read `src/train.py` for command-line options, then `training.py` for the sequence of operations. Training contacts update the parameters; validation AUPRC chooses the epoch; the test set is read after that choice. Checkpoint loading and inference helpers are imported from their own modules so the training loop can be read on its own.

The demonstration datasets and commands are described in [TRAINING.md](TRAINING.md). Running an import or reading this repository does not start training.

## Original implementations


Prepared inputs, labels and source tables live in `data/`. Dataset construction and preprocessing scripts are not part of the release.

## Full benchmark and figures

`src/benchmark.py` reads the supplied PT inputs and NPZ labels. It supports multi-vector, sequence-only and single-vector checkpoints using the adjacent configuration. PPI duplicate accession pairs use maximum score and logical-OR labels; PRI exact-entity groups use maximum score and the provided cohort masks. `--scores` accepts a frozen PRI exact-entity NPZ for a fast mask/metric check.

`src/plot_figures.py` provides numerical replots and exports the final editable SVG compositions. It does not run training or change scoring parameters. Official SVG compositions are frozen presentation assets, not a substitute for the separately supplied numerical tables.
