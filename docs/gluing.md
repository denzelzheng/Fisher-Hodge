# Gluing thermodynamic manifolds

This note states when two fitted charts can be joined along shared species, what the join needs, and what it cannot know.

## Setting

Module $A$ has species $(X,Y)$ and stoichiometry $S_A$. Module $B$ has species $(Y,Z)$ and stoichiometry $S_B$. The species $Y$ are shared. The composite has species $(X,Y,Z)$ and $S=[\,S_A\ \ S_B\,]$, with each block extended by zero on the species its module does not have.

Each module has its own fitted $\mu^0$, excess free energy $G$ and mobility $k$. The composite needs one potential field, one free energy, and a mobility for each hyperedge.

## Part 1. Potentials: the obstruction is the set of new cycles

A fitted $\mu^0$ is defined up to the conservation laws of its module. The shift $\mu^0\to\mu^0+L^\top\lambda$ with $L$ a basis of $\ker S^\top$ changes neither the force $f$ nor the factor $\varepsilon$, since $\varepsilon$ reads only $\log c+\nabla G$. Gluing asks for shifts that make the two modules agree on $Y$.

**Theorem 1.** Let $C=\ker S$ be the cycles of the composite and let $C_A\oplus C_B$ be the cycles of the modules. Then

$$C/(C_A\oplus C_B)\;\cong\;\operatorname{im}S_A\cap\operatorname{im}S_B\;\cong\;\big(L_A|_Y+L_B|_Y\big)^\perp\subset\mathbb R^{Y}.$$

Module potentials with forces $f_A=S_A^\top\mu_A$ and $f_B=S_B^\top\mu_B$ come from one global potential if and only if $\langle j,(f_A,f_B)\rangle=0$ for every $j\in C$.

*Proof.* A vector $(j_A,j_B)$ lies in $C$ exactly when $S_Aj_A=-S_Bj_B$. The map sending it to this common value has image $\operatorname{im}S_A\cap\operatorname{im}S_B$ and kernel $C_A\oplus C_B$, which gives the first isomorphism. A vector in $\mathbb R^{Y}$, extended by zero, lies in $\operatorname{im}S_A$ exactly when it is orthogonal to $\ker S_A^\top$, which means orthogonal to $L_A|_Y$. The same holds for $B$, which gives the second. The forces $(f_A,f_B)$ lie in $\operatorname{im}S^\top$ exactly when they are orthogonal to $\ker S$. On $C_A\oplus C_B$ this orthogonality already holds, so only $C/(C_A\oplus C_B)$ remains. $\square$

The first isomorphism is a Mayer–Vietoris sequence for the two stoichiometries. The condition on forces is the Wegscheider condition of the composite.

Three consequences follow.

- If gluing creates no new cycle, alignment is always possible and the module forces are unchanged.
- If gluing creates new cycles, neither module constrained the potential around them. Errors in the fitted potentials then appear as an affinity around a new cycle that has no physical drive behind it.
- The residual of the alignment is this spurious affinity, one number per new cycle. Projecting onto one potential field removes it and changes the module forces by the same amount.

A chart therefore reports, for each new cycle, the spurious affinity before projection. Known chemostat free energies fix the true drive around cycles through chemostats, so they do not enter this residual.

## Part 2. Free energy: inclusion–exclusion

Let $G$ be the excess free energy of the composite. Module $A$ observes $G(x,y,0)$ and module $B$ observes $G(0,y,z)$.

**Theorem 2.** The identity

$$G(x,y,z)=G(x,y,0)+G(0,y,z)-G(0,y,0)$$

holds for all states if and only if $\partial_x\partial_z G=0$ for all states.

*Proof.* Let $D$ be the left side minus the right side. Then $\partial_x\partial_z D=\partial_x\partial_z G$, because the terms on the right depend on at most one of $x,z$. If this vanishes then $D(x,y,z)=a(x,y)+b(y,z)$, and $D(x,y,0)=D(0,y,z)=0$ gives $D=0$. The converse is immediate. $\square$

The term $G(0,y,0)$ is the self energy of the shared species, which both modules contain. This is the same decomposition as $\log p_{AB}=\log p_A+\log p_B-\log p_S$ for a Markov chain $X-Y-Z$, with free energy in the place of the negative log density. The condition $\partial_x\partial_zG=0$ also excludes three-body terms such as $xyz$.

Each module gives its own estimate of the self energy, $G_A(0,y)$ and $G_B(y,0)$. We use their mean, and report their difference as a second diagnostic. This difference also flags extrapolation, since a module may never have seen $c_X$ near zero.

## Part 3. What data of the modules cannot determine

**Proposition 3.** Let $G^\ast$ and $k^\ast$ be any truth. For every function $h(x,y,z)$ that vanishes when $x=0$ or $z=0$, the excess free energy $G^\ast+h$ produces the same trajectories on the data of both modules. So does a mobility on the hyperedges of $A$ multiplied by $e^{w(x,y,z)}$ with $w$ vanishing at $z=0$, and the same for $B$.

*Proof.* Module $A$ has $z=0$ throughout and no hyperedge through $Z$. On the plane $z=0$ the modified functions equal the truth, and so do their derivatives along $x$ and $y$, which is all the dynamics of $A$ read. Module $B$ is the same with $x=0$. $\square$

The cross terms are therefore outside what any method can learn from the modules alone. Two kinds of cross term exist. One is thermodynamic, in $G$. The other is kinetic, in $k$, as in inhibition of a step in $A$ by a species of $B$.

## Zero-shot and refined composition

**Corollary 4.** Suppose the truth satisfies $\partial_x\partial_zG=0$, the mobility of each module does not read the species of the other, and each module is fitted exactly on the states the composite visits. Then the glued chart gives the true rates of the composite, up to the shift of Theorem 1.

Reaction joined with ideal transport meets the first two conditions. The last condition is about coverage. The composite can visit states that neither module saw, and the error then depends on how the networks extrapolate. With fitted modules the guarantee becomes a bound, in which the force error on a hyperedge comes from the error of its module and the projection of Theorem 1.

When the first two conditions fail, the composite needs data. A short refinement keeps the modules fixed and fits only

$$G_{\rm cross}(x,y,z),\qquad g_{\rm cross}(h,\,x,y,z),$$

both zero at the start and both vanishing where Proposition 3 says the modules already determine the answer. Local detailed balance and $\sigma\ge0$ with respect to the model's force hold for any such terms.

## What gluing preserves

At equal potentials, $\sigma=\sigma_A+\sigma_B$. Local detailed balance holds on every hyperedge. The conservation laws of the composite are $\ker S^\top$, which is usually smaller than either module's set. In the coordinates of the chart, the composite manifold is the product of the module manifolds restricted to one potential field.

## Related work

Baez and Pollard ([2017](https://doi.org/10.1142/s0129055x17500283)) give a compositional framework for reaction networks, Baez, Fong and Pollard ([2016](https://doi.org/10.1063/1.4941578)) do the same for Markov processes, and Baez, Lynch and Moeller ([2023](https://doi.org/10.1063/5.0089375)) treat equilibrium thermostatics. Those works concern known systems. The note above concerns fitted ones.

Learning parts and predicting the whole has been done before. Neary and Topcu ([2022](https://arxiv.org/abs/2212.00893)) train port-Hamiltonian networks on subsystems, predict composite systems without composite data, bound the composite error, and learn the interconnection when it is unknown. Hernández et al. ([2023](https://doi.org/10.1007/s00466-023-02296-w)) learn complex systems by parts with port-metriplectic networks, which satisfy the laws of thermodynamics by construction. In those models subsystems meet at ports. Here modules meet at shared species of a hypergraph, which is what brings in the potential gauge, the new cycles, and the double-counted self energy.

## Appendix: cases with cross terms

Both cases use the modules of the two-cycle benchmark, so the modules are the same in all three tables and only the composite truth differs. The design fixed before any result says that both should fail without refinement (`benchmarks/composition/__init__.py`). A refinement fits only the cross terms, for 500 steps, on coupled trajectories of 20 steps. Errors are the relative L2 error of the state at 10T, from one seed.

**Thermodynamic cross term,** $G=\chi c_Xc_Y$ with $\chi=0.5$, so that $\partial_x\partial_zG\neq0$.

| Method | training conditions | stronger drive | start × 1.5 |
|---|---|---|---|
| **Glued chart, no retraining** | 0.029 | 6.0e-03 | 0.050 |
| Glued chart, refined on 8 | 4.3e-03 | 2.4e-03 | 2.8e-03 |

**Kinetic cross term.** The rate constant of the first step is divided by $1+c_{Y_p}/K_I$ with $K_I=0.5$.

| Method | training conditions | stronger drive | start × 1.5 |
|---|---|---|---|
| **Glued chart, no retraining** | 0.135 | 0.096 | 0.199 |
| Glued chart, refined on 8 | 0.011 | 6.5e-03 | 0.020 |

Without refinement the glued chart fails in both cases, as Proposition 3 predicts. With 8 coupled trajectories the error falls by a factor of 2.5 to 18 in the thermodynamic case and of 10 to 15 in the kinetic case.
