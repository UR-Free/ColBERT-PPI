# Training protein identifiers

`ppi_uniprot.csv` and `pri_uniprot.csv` list the unique UniProt accessions recorded for proteins in the training inputs. They contain identifiers only, not the full training sequences, structures or contact annotations. `pri_unmapped.csv` lists source records lacking a recorded accession; an empty table means all records were mapped.

The PRI accessions identify the AlphaFold/UniProt protein source used to prepare each input. They do not assert that a complete UniProt sequence is identical to the experimental chain fragment. These identifier lists are not a substitute for the annotated training pairs required to reproduce full model training. Ten annotated training examples per task are provided separately in `examples/training/`.

Evaluation inputs in `data/ppi/` and `data/pri/` are already processed. No preprocessing scripts are distributed.
