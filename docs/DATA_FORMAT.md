# Data format

The repository contains no biological data. Users prepare the following local
artifacts and reference them from a JSON configuration.

## Preparation manifest

`colbert-ppi-prepare-data` accepts a CSV with four required columns:

```csv
protein_a,protein_b,structure_a,structure_b
complex01_A,complex01_B,structures/complex01_A.pdb,structures/complex01_B.pdb
```

Structure paths may be absolute or relative to the manifest. Each PDB must
contain one protein chain. Both structures in a row must retain the coordinate
frame of the same complex so their inter-chain distances remain meaningful.
The command uses Foldseek for 3Di extraction and writes the pair CSV, tokenized
SaProt inputs, Cβ coordinate archive and sparse contact labels described below.

## Pair CSV

The loader accepts a header containing one pair identifier column. A minimal
example is:

```csv
PAIRID
protein_a:protein_b
```

Identifiers must resolve to keys in both the SaProt-input and coordinate
artifacts. PINDER-style identifiers may include chain and UniProt aliases.

## SaProt inputs

A PyTorch file maps protein identifiers to dictionaries with:

- `input_ids`: integer tensor of shape `(1, L)`;
- `attention_mask`: integer or Boolean tensor of shape `(1, L)`.

Special tokens are included. Training truncation is applied in token space.

## Cβ coordinates

An NPZ archive contains:

- `labels`: protein identifiers;
- `offsets`: start offsets into the concatenated coordinate array;
- `coords`: float array of shape `(N, 3)`.

## Sparse contact labels

A PyTorch dictionary is keyed by the exact pair identifier
`protein_a:protein_b`. Each value contains sparse residue-index pairs:

- `pos_i`, `pos_j`: labelled contacts;
- `neg_i`, `neg_j`: labelled non-contacts.

Indices are zero-based and must already be projected into the residue
coordinates used by the corresponding monomer inputs. Unlabelled residue pairs
are ignored.

## Retrieval label matrix

`colbert-ppi-evaluate` reads a Boolean matrix from an NPZ archive. By default,
the array key is `labels`; use `--label-key` to select another key. For a split
with \(N\) dataset records, the matrix must have shape `(N, N)` and follow the
exact CSV record order. Entry `(i, j)` indicates whether receptor-side record
`i` and ligand-side record `j` form a positive interaction.

## Privacy

Do not place access tokens, human-subject identifiers, private server paths or
licensed third-party databases in the repository. Store datasets and model
weights outside Git and reference them through relative configuration paths.
