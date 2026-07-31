# Data format

The repository contains no biological data. Users prepare the following local
artifacts and reference them from a JSON configuration.

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
