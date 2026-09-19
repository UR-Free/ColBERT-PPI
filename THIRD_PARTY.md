# Third-party materials

The repository's code licence does not relicense external pretrained models,
software or underlying datasets. Obtain backbones from their original sources
and observe the terms distributed with them.

| Resource | Role | Upstream source |
|---|---|---|
| SaProt / SaProt_650M_PDB | Protein backbone and token vocabulary | https://github.com/westlake-repl/SaProt ; https://huggingface.co/westlake-repl/SaProt_650M_PDB |
| ERNIE-RNA | Optional RNA backbone, dictionary and code | https://github.com/Bruce-ywj/ERNIE-RNA |
| Foldseek | Structure-to-3Di preprocessing | https://github.com/steineggerlab/foldseek |
| PINDER | Source of processed PPI examples and benchmark inputs | https://github.com/pinder-org/pinder |
| UniProt | Protein accession identifiers | https://www.uniprot.org/ |

PyTorch, Transformers, PEFT, NumPy, SciPy and other dependencies retain their
own licences. They are installed as dependencies rather than copied into this
repository. External backbone weights and the ERNIE-RNA implementation are not
redistributed in release assets.

Processed benchmark artifacts retain source identifiers and cohort definitions.
Refer to `docs/DATA_GUIDE.md`, the source tables and the manuscript for dataset
provenance. Comparator predictions are frozen evidence, not redistributions
of comparator software or model weights. Author-created code, learned
components and derived tables do not change rights in underlying source data.
