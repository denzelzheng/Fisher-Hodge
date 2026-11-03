# Appendix: what information geometry contributes

The README uses a few plain geometric terms: the thermodynamic manifold, the chart, the driving coordinate $f$, the material coordinate $k$, and the boundary faces $k_e=0$.
This note says what the underlying information geometry does for the method.
The statements and proofs are in the section "Information geometry of the chart" of [theory.md](theory.md) (IG1 to IG7).

## 1. Why $(f,k)$ are good coordinates

The flux of a step depends nonlinearly on its force, $J=g\cdot2\sinh(f/2)$ with $f=\log(a/b)$ and $g=\sqrt{ab}$.
At a frozen state the forward and backward event counts are Poisson laws with intensities $a$ and $b$, and these laws form an exponential family.
Its natural coordinates are $(f,\log g)$, and its dual coordinates are the current and the traffic $a+b$ (IG2(a)).
The family is dually flat.

In the chart $\log g=\log k+\log\varepsilon(c,\theta)$, and at a fixed state both $f$ and $\log\varepsilon$ are affine in the natural parameters of the conditions (IG2(b)).
A change of condition is therefore a translation by a known vector in the natural coordinates.
Model and truth differ by the vector $(0,\log(k/k^{\rm true}))$ at every condition.
This gives the factorization $g=k\varepsilon$ its meaning: $k$ is the part of the natural coordinates that a change of condition does not move.

The two coordinates are decoupled in this product sense.
They are Fisher-orthogonal only when the current vanishes, since the Fisher metric of a step in $(f,\log g)$ has off-diagonal entry $J/2$ (IG2(a)).

## 2. Why the start is a convex problem

With $f$ and $\varepsilon$ fixed, the rates $a=k\varepsilon e^{f/2}$ and $b=k\varepsilon e^{-f/2}$ are linear in $k$.
The set of laws reachable by varying $k\ge0$ is then a convex cone in the dual coordinates, and the equation-error fit is the Euclidean projection of the data onto the image of that cone.
This projection is convex and has a global optimum (Theorem 0(iii)(c)).
It uses the Euclidean metric of the equation error, so it is not a divergence projection.
It also covers only this step. The mapping back to coupled parameters, the fit of the activity correction and the rollout training are not convex.

## 3. What it says about dissipation and transfer

The entropy production of a step is the Jeffreys divergence between its law and the reversed law, and summed over steps it is the path divergence between the process and its time reversal (IG1).
Each Jeffreys term is a sum of two Bregman divergences of the Poisson family.
The Onsager metric that appears in $\sigma=\|f\|_\Lambda^2$ is the Fisher information averaged along the geodesic from $a$ to $b$ (IG3).

In that metric the entropy production splits exactly into an excess part and a housekeeping part, $\sigma=\sigma_{\rm ex}+\sigma_{\rm hk}$, by a Pythagorean relation (IG4(a)).
The housekeeping part is carried by the fluxes in the cycle space $\ker S$, and it is the part that the cycle affinities drive (Theorem 1).
At a steady state all of $\sigma$ is housekeeping.
This is the structure behind the readouts of cycle affinity and entropy production in the phosphorylation cycle.

The same metric controls the error at a new condition.
At a fixed state $\|J-J^{\rm true}\|^2_{\Lambda^{-1}}=\sum_h\delta_h^2\sigma_h^{\rm true}$ with $\delta_h=k_h/k_h^{\rm true}-1$ (IG2(c)).
Over a trajectory the error also depends on the Lipschitz constant of the dynamics at that condition, through Gronwall's lemma (Theorem 0(iii)(b)).

## 4. Where identifiability fails

For density data with noise of intensity $\eta^2$, the Fisher information of $k$ is $X^\top X/\eta^2$, and its null space carries no information at any sample size (IG7(b)).
Catalytic variants and split half-reactions on a library of candidates are directions of this null space, or close to it (the parallel-channel lemma of Theorem 0(iii)(d)).
A nearly irreversible step is a second source: its efficient information tends to zero with its reverse rate (IG7(a)).
