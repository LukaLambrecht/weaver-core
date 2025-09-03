# Study the effect of training on synthetic data instead of simulation

Comes on top of baseline + resampling + extra signal samples.
The signal samples and synthetic data have been ntupled using FeynNet for pairing the jets,
rather than the default MinDiag method, to see how it affects the background sculpting.

This study uses input files in which the objects (notably jets) have already been pre-selected,
to avoid a potential mismatch between masking and proper selection.
