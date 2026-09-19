# Small training datasets

Each task includes 10 training pairs, 2 validation pairs and 2 test pairs. All examples are real annotated complexes. The same protein or RNA endpoint does not occur in more than one mini split. Complete token sequences are retained; no contact-centred cropping was used.

PPI examples were selected deterministically from the original PINDER training input pool and partitioned into separate mini splits. Thus these are demonstration splits, not the manuscript PINDER validation/test cohorts. PRI examples come from their respective original train, validation and test pools. Selection uses short complete chains and requires positive and negative contact annotations; it is independent of model scores.

Token arrays contain BOS and EOS. Contact indices are zero-based and exclude special tokens. Positive pairs are experimentally annotated contacts; negative pairs are labelled noncontacts. These contact negatives are distinct from the operational protein-pair negatives used for retrieval evaluation.

These samples are sufficient to exercise optimisation, checkpoint selection, held-out evaluation and inference. Metrics from two held-out pairs are smoke-test outputs and must not be compared with the manuscript's full benchmark results.
