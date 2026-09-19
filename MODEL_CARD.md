# Model card

ColBERT-PPI ranks candidate protein partners using structure-aware SaProt
representations and a contact-supervised residue encoder. This release also
includes protein–RNA (PRI) transfer models.

## Inputs and outputs

PPI consumes SaProt amino-acid/3Di tokens from individual protein structures.
Both computational roles are encoded explicitly. The default checkpoint is
epoch 69, paired with its own fixed training-reference bank. PRI consumes
protein tokens and ERNIE-RNA nucleotide tokens. See [input preparation](docs/INPUTS.md)
and [model settings](docs/VERSION.md).

Outputs are relative retrieval scores, **not calibrated probabilities**, binding
 affinities, or proof of interaction. Higher values indicate stronger matching
 evidence within the same task, model and input protocol. There is no released
 universal binary decision threshold. Compare scores only under matching
 checkpoint, reference bank, preprocessing and scoring settings.

## Intended use and limitations

The intended use is computational research and prioritisation of experiments.
Predicted interactions require independent experimental validation. Structure
quality, missing residues, chain choice, sequence length and training-set
relatedness can affect results. Inputs longer than 1,024 represented residues
must be explicitly split or cropped; preparation never truncates silently.
Residue indices refer to represented structure positions, which may differ
from full-length UniProt numbering.

PINDER retrieval negatives are operational database-filtered negatives, not
necessarily experimentally demonstrated non-interactions. Unjudged pairs are
excluded from benchmark evaluation. Localisation uses the raw cosine matrix;
its values do not establish biochemical causality.

The CPU example verifies the readout against frozen vectors. Small training
examples verify plumbing and are not benchmark evidence. Full original
training-data reconstruction is not provided. See [contents](docs/CONTENTS.md)
and [data guide](docs/DATA_GUIDE.md).

## Dependencies and provenance

SaProt_650M_PDB is an external required backbone; PRI additionally requires
ERNIE-RNA. These pretrained models and third-party data retain upstream terms.
Author-trained components omit frozen backbone weights and optimizer states.
Asset SHA-256 checksums identify the distributed bytes.
