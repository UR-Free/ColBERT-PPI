# Model notes

ColBERT-PPI is intended for ranking candidate molecular partners and examining residue-level evidence. Its scores are not calibrated probabilities of physical binding. The localisation readout does not reconstruct a three-dimensional complex or establish a binding mechanism.

The PPI model retains a vector for each residue and uses role-conditioned contextual encoding. During retrieval, both assignments of query and candidate roles are evaluated. Contact supervision shapes the protein representation; a fixed training-reference correction and smooth aggregation are applied to retrieval scores. Interface localisation uses the uncorrected residue matrix.

For protein–RNA adaptation, the PPI-trained protein branch is paired with ERNIE-RNA and trained with protein–RNA supervision. Single-vector controls are trained and evaluated separately with their native pooled scoring function.


Test benchmarks use their stated operational-negative definitions. Performance depends on candidate composition and label prevalence. The same evaluation labels must be used when comparing models.
