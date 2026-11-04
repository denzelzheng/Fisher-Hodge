# Notes

The phosphorylation cycle on heterogeneous reaction media. Every medium is a hidden periodic map H(x), with 8 pixels per site, standard deviation 1, wavenumbers up to 2n/8 (70 % of the variance) and 6n/8 (30 %).

## Truth

The symmetric rate constant of every hyperedge is multiplied by a function of H and does not read the conditions:

- R1 at a site: the mean over the pixels of the cell of exp(1.2 tanh(1.5 H)).
- R2 at a site: the mean over the pixels of the cell of exp(1 − 2 exp(−H²)).
- Transport of X: the harmonic mean of exp(0.8 tanh(1.5 H)) over the 8 pixels between the two cell centres.
- Transport of Xp: the harmonic mean of exp(−0.6 tanh(1.5 H)) over the same pixels.

## Inputs

The models see 3 descriptors per hyperedge. On a reaction: the mean of H in the cell, its standard deviation in the cell, and the mean of the cell means over the 4 neighbours. On a transport hyperedge: the mean of the cell means at the two ends, the mean of the cell standard deviations, and the absolute difference of the cell means. CRNN (mass action) has a rate table per class and takes no descriptors.

## Test sets

All on unseen media: training conditions, drive ΔG 3–4, a stronger and finer potential U, and the 16×16 grid with the same spacing. A finer mesh at fixed domain would change D / h², which is not an input of these models, so the zero-shot grid is a larger lattice.

## Integrator tolerance

The oracle's median 10T error is 1.7e-6 to 2.1e-6 at every rtol from 1e-2 to 1e-5 on the sets with ΔG 1–3, and 5e-7 with ΔG 3–4. The benchmark uses rtol 1e-4.
