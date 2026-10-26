# HyperME theory notes

These notes state four results that the HyperME experiments already support numerically. Each item gives a statement, a proof or proof sketch, a label that separates known results from statements of this project, and one line of numerical evidence.

## 0. Setting and notation

There are $n$ state coordinates (species at sites) and $m$ hyperedges. Hyperedge $h$ has reactant and product vectors $\nu_h^+,\nu_h^-\in\mathbb{Z}_{\ge 0}^n$, and $S=\nu^--\nu^+\in\mathbb{R}^{n\times m}$. The rates are

$$a_h=k_h\,e^{\langle\nu_h^+,\mu\rangle+A_h/2},\qquad b_h=k_h\,e^{\langle\nu_h^-,\mu\rangle-A_h/2},\qquad J=a-b,\qquad \dot c=SJ,$$

with $k_h>0$ and $\mu=\partial F/\partial c$. The edge force is $f_h=\log(a_h/b_h)=-\langle S_h,\mu\rangle+A_h$. With the logarithmic mean $\ell_h=\Lambda(a_h,b_h)=(a_h-b_h)/(\log a_h-\log b_h)$ the Onsager relation $J=\ell\odot f$ holds exactly, and $\sigma=\sum_h J_hf_h=\sum_h\ell_hf_h^2\ge 0$.

**Gauge lemma.** For any $\varphi\in\mathbb{R}^n$ the map $(\mu,A,k_h)\mapsto(\mu+\varphi,\ A+S^\top\varphi,\ k_h e^{-\langle\nu_h^++\nu_h^-,\varphi\rangle/2})$ leaves every $a_h$ and $b_h$ unchanged. *Proof.* Both exponents gain $\langle\nu_h^++\nu_h^-,\varphi\rangle/2$, which the new $k_h$ cancels. $\square$ So only the class $[A]\in\mathbb{R}^m/\operatorname{im}S^\top$ is meaningful unless the potential part of $\mu$ is fixed by other means.

We write $\beta_1:=\dim\ker S=m-\operatorname{rank}S$. For a graph incidence matrix $\operatorname{rank}S=|V|-c$, so $\beta_1=|E|-|V|+c$, the first Betti number. For a hypergraph we use $\beta_1$ as the definition, through the two-term chain complex $\mathbb{R}^m\xrightarrow{S}\mathbb{R}^n$ with $H_1=\ker S$ and $H_0=\operatorname{coker}S$. The dimension of $H_0$ counts the conservation laws.

## Theorem 1. Minimal observables and the blind spot of densities

**Statement.** Let $c(t)$ be observed exactly on an interval.

(a) The flux histories compatible with $c(t)$ form the affine set $\{J(t)+z(t): z(t)\in\ker S\}$.

(b) $\sigma=-\langle\dot c,\mu\rangle+\langle J,A\rangle$. For fixed forces, the shift $J\to J+z$ with $z\in\ker S$ changes $\sigma$ by $\langle z,A\rangle$. This depends on $A$ only through $[A]$, and the pairing $z\mapsto\langle z,A\rangle$ identifies $[A]$ with an element of $(\ker S)^*$. The cycle affinities $\langle z,f\rangle=\langle z,A\rangle$, $z\in\ker S$, are this element.

(c) Add linear observations $y_r=\langle w_r,J\rangle$, $r=1,\dots,R$. The unresolved flux directions form $\ker S\cap\bigcap_r\ker w_r$, of dimension $\beta_1-\dim\big((\operatorname{span}\{w_r\}+\operatorname{im}S^\top)/\operatorname{im}S^\top\big)$. A new observable lowers this number by exactly one if $w\notin\operatorname{im}S^\top+\operatorname{span}\{w_1,\dots,w_{R}\}$, and by zero otherwise. Two observables are equivalent when $w-w'\in\operatorname{im}S^\top$. The currents on any $\beta_1$ "chord" hyperedges, whose complement is a column basis of $S$, determine $J$ completely.

(d) (Hodge split.) For any positive weight $\ell$,
$$\mathbb{R}^m=\operatorname{im}S^\top\oplus_\ell\ \ell^{-1}\ker S$$
is orthogonal in $\langle f,g\rangle_\ell=\sum_h\ell_hf_hg_h$. Equivalently, in flux space with $\langle J,J'\rangle_{\ell^{-1}}$, $\mathbb{R}^m=\ell\,\operatorname{im}S^\top\oplus\ker S$. Writing $f=f_{\rm ex}+f_{\rm hk}$ gives $\sigma=\sigma_{\rm ex}+\sigma_{\rm hk}$ with
$$\sigma_{\rm ex}=\dot c^\top(S\ell S^\top)^+\dot c=\min\{\|J'\|^2_{\ell^{-1}}:SJ'=\dot c\},\qquad J_{\rm hk}=\ell f_{\rm hk}\in\ker S .$$
For given $\ell$, $\sigma_{\rm ex}$ is fixed by $c(t)$. The housekeeping part is the squared norm of a flux in $\ker S$, so it lies entirely in the blind spot of (a).

(e) $[A]=0$ if and only if a detailed-balanced equilibrium exists. For graphs this is the Kolmogorov criterion, and for reaction networks it is the Wegscheider condition.

**Proof.** (a) $\dot c$ determines $SJ$, and $SJ=SJ'$ iff $J-J'\in\ker S$. (b) Insert $f=-S^\top\mu+A$ into $\sigma=\langle J,f\rangle$ and use $\langle J,S^\top\mu\rangle=\langle SJ,\mu\rangle$. For $z\in\ker S$ and any $\varphi$, $\langle z,A+S^\top\varphi\rangle=\langle z,A\rangle$, so the pairing is well defined on classes. It is injective on $\mathbb{R}^m/\operatorname{im}S^\top$ because $(\ker S)^\perp=\operatorname{im}S^\top$. (c) Observing $y_r$ fixes the restriction of $w_r$ to $\ker S$. A functional vanishes on $\ker S$ iff it lies in $\operatorname{im}S^\top$, and then $\langle S^\top\varphi,J\rangle=\langle\varphi,\dot c\rangle$ is already known. For chords $C$ and basis columns $B$, $S_BJ_B=\dot c-S_CJ_C$ has a unique solution because the columns of $S_B$ are independent and span $\operatorname{im}S$. (d) $\langle S^\top\varphi,\ell^{-1}z\rangle_\ell=\langle\varphi,Sz\rangle=0$, and the dimensions add to $m$. With $f_{\rm ex}=S^\top\psi$, $S\ell S^\top\psi=SJ=\dot c$, which gives the formula for $\sigma_{\rm ex}$. Any $J'$ with $SJ'=\dot c$ splits $\ell^{-1}$-orthogonally as $\ell S^\top\psi+z$ with $z\in\ker S$, which gives the minimum. (e) If $A=S^\top\varphi$, then $\mu^*=\varphi$ gives $f=0$, and with $\mu=\log c+U/T$ this $\mu^*$ is attained by $c=e^{\mu^*-U/T}>0$. Conversely $f(\mu^*)=0$ gives $A=S^\top\mu^*$. By (b), $[A]=0$ iff $\langle z,A\rangle=0$ for all $z\in\ker S$. For a graph, simple cycles span $\ker S$ and $\langle z,A\rangle$ is the log ratio of the products of forward and backward rates around $z$. $\square$

**Model-level realization (Proposition 1').** Part (a) does not use the rate law. A parametric rate law can resolve part of $\ker S$. For a linear Markov chain on a simple graph with rich initial data, $\dot\rho=\rho Q$ identifies $Q$, and hence $[A]$. In that case the information comes from the kinetic assumption, not from the densities. The blind spot is realized inside HyperME when a cycle consists of kinetically indistinguishable channels. *Example (C1).* After the chemostatted species are removed, R1 is $X\to X_p$ and R2 is $X_p\to X$. Then $\dot c_{X_p}=\alpha e^{\mu_X}-\beta e^{\mu_{X_p}}$ with $\alpha=k_1e^{A_1/2}+k_2e^{-A_2/2}$ and $\beta=k_1e^{-A_1/2}+k_2e^{A_2/2}$. Fix $\alpha,\beta$ and any cycle affinity $C=A_1+A_2>0$. Solving for $(k_1,k_2)$ gives positive values whenever $\log(\alpha/\beta)<A_1<\log(\alpha/\beta)+C$, and this interval is never empty. So every $C$ fits the same noiseless trajectories, and $\sigma$ is unbounded above on the fitted set. $\square$

**Label.** (a), (b) and (e) are known: Schnakenberg (1976), Jiang, Qian and Qian (2004), and Rao and Esposito (2016) for $\sigma=-\dot F+\langle J,A\rangle$. (d) is the $\ell$-weighted orthogonal decomposition of Kolchinsky, Dechant, Yoshimura and Ito (arXiv:2206.14599, 2022) and Yoshimura et al. (PRR 2023). Its Hessian-geometric form for reaction networks is due to Kobayashi, Loutchko, Kamimura and Sughiyama (PRR 4, 033208, 2022). Statements of this project: the observable count in (c), used as an experimental-design rule, and Proposition 1'. Both are elementary.

**Evidence.** On C1, $\dim\ker S=257$. One global ATP turnover leaves 256 invisible directions, and four local currents leave 253. $\sum_vJ_{R1}$ and $\sum_vJ_{R2}$ are equivalent, with residual $2\times10^{-14}$ in $\operatorname{im}S^\top$.

## Theorem 2. Cramér–Rao bound for the affinity of a near-irreversible hyperedge

**Statement.** Take one hyperedge whose forward and backward events are counting processes with intensities $a(t)$ and $b(t)$ on $[0,T]$, where $\log a=u+x(t)+A/2$, $\log b=u+y(t)-A/2$, $u=\log k$, and $x=\langle\nu^+,\mu\rangle$, $y=\langle\nu^-,\mu\rangle$ are known from the observed concentrations. Let $\bar N_+=\mathbb{E}\int_0^Ta\,dt$ and $\bar N_-=\mathbb{E}\int_0^Tb\,dt$.

(i) If $k$ is known, $I(A)=(\bar N_++\bar N_-)/4$.

(ii) If $k$ is a nuisance, the efficient information is
$$I_{\rm eff}(A)=\frac{\bar N_+\bar N_-}{\bar N_++\bar N_-}\le\min(\bar N_+,\bar N_-),\qquad \operatorname{Var}(\hat A)\ge\frac1{\bar N_+}+\frac1{\bar N_-}.$$
For constant rates and $b\ll a$ this is $\operatorname{Var}(\hat A)\gtrsim 1/(bT)$. The constant is 1, not 1/4. The factor 1/4 belongs to case (i), where it multiplies $a+b$ and the bound is mild.

(iii) The bound in (ii) holds for any data that are a function of the network jump record, including concentration trajectories, provided $(A_h,k_h)$ are parameters of hyperedge $h$ alone. Further nuisance parameters only lower $I_{\rm eff}$.

(iv) A standard error $\varepsilon$ requires $\bar N_-\ge(1+\bar N_-/\bar N_+)/\varepsilon^2$, that is, about $\varepsilon^{-2}$ backward events. For constant rates with $b=ae^{-A}$, this is $aT\ge(1+e^{A})/\varepsilon^2$. With extensive propensities $a=\Omega\hat a$, it is
$$\Omega T\ \ge\ \frac{1+e^{A}}{\hat a\,\varepsilon^2}.$$
The cost grows exponentially in the affinity.

(v) (Deterministic data with noise.) If the flux $J=a-b$ is observed with white noise of intensity $\eta^2$ and $k$ is a nuisance, then
$$I_{\rm eff}(A)=\eta^{-2}\Big[\int b^2\,dt-\frac{(\int bJ\,dt)^2}{\int J^2\,dt}\Big]\le\eta^{-2}\int_0^T b^2\,dt .$$
If $b/a$ is constant in time, $I_{\rm eff}=0$.

**Proof.** For a counting process with predictable intensity $\lambda_\theta$, the Fisher matrix is $\mathbb{E}\int\lambda\,\nabla_\theta\log\lambda\,\nabla_\theta\log\lambda^\top dt$ (Andersen, Borgan, Gill and Keiding 1993). With $\partial_A\log a=\tfrac12$, $\partial_A\log b=-\tfrac12$ and $\partial_u\log a=\partial_u\log b=1$,
$$I=\begin{pmatrix}(\bar N_++\bar N_-)/4&(\bar N_+-\bar N_-)/2\\(\bar N_+-\bar N_-)/2&\bar N_++\bar N_-\end{pmatrix}.$$
Its $(A,A)$ entry gives (i). The Schur complement gives $[(\bar N_++\bar N_-)^2-(\bar N_+-\bar N_-)^2]/[4(\bar N_++\bar N_-)]$, which is (ii). For (iii), the network path likelihood is a sum over channels, so the Fisher matrix of the full record is block diagonal and the $(A_h,u_h)$ block equals (i)–(ii) along the path. A statistic of the record has a smaller Fisher matrix in the Loewner order, and the Schur complement is monotone in that order. (iv) follows from (ii). For (v), the sensitivities are $p=\partial_AJ=(a+b)/2$ and $q=\partial_uJ=J$. With $a+b=J+2b$, the Schur complement $\eta^{-2}[\int p^2-(\int pq)^2/\int q^2]$ reduces to the stated form. $\square$

**Remarks.** In (ii) the information rate is $ab/(a+b)$, half the harmonic mean. It never exceeds $\ell=\Lambda(a,b)$, which is itself the Poisson log-rate Fisher information averaged along the e-geodesic $a^{1-s}b^s$. An error $\delta A$ shifts the edge contribution $\sigma_h=J_hf_h$ by $J_h\delta A$. So the near-irreversible edge limits the housekeeping readout of Theorem 1(d) to precision $J_h/\sqrt{\bar N_-}$. This links to hidden entropy production in two ways. Esposito (2012) shows that unresolved processes hide entropy production structurally under coarse graining. Here the backward channel is present in the model, but it carries too few events, so the loss is statistical. In the decomposition of Kolchinsky et al. (2022), the housekeeping part is the projection onto nonconservative forces, and for a near-irreversible cycle it is dominated by that cycle's affinity, so it inherits the bound. The limit $b\to0$ is the absolutely irreversible case of Murashita, Funo and Ueda (2014).

**Label.** The Fisher matrix of counting processes is known. Statements of this project: (ii)–(v) in the HyperME parametrization, especially the role of $k$ as a nuisance and the Gaussian form (v).

**Evidence.** On C1, C3 and E5 with backward rates of $10^{-3}$ to $10^{-2}$, the learned affinity reaches only 15–40% of the true value. In E5, $\log(k_2/k_2')=3.9$ is recovered as 0.6–1.5, and in WS3, 4.91 is recovered as 1.43.

## Theorem 3. Closed-form coarse graining

**Setting.** Here $Q$ is an irreducible generator with $Q_{ij}>0\Leftrightarrow Q_{ji}>0$. Distributions are row vectors with $\dot p=pQ$, and $\pi Q=0$. For a partition into blocks $a$, let $P_{ia}=\mathbf 1[i\in a]$, $\bar\pi=\pi P$, and $\Pi_{ai}=\pi_{i|a}=\pi_i/\bar\pi_a$ for $i\in a$, so $\Pi P=I$. Define $\bar Q=\Pi QP$, that is, $\bar Q_{ab}=\sum_{i\in a}\pi_{i|a}\sum_{j\in b}Q_{ij}$, and $J_{ij}=\pi_iQ_{ij}-\pi_jQ_{ji}$. In the HyperME form for a linear chain, $Q_{ij}=k_{ij}e^{(\mu_i-\mu_j)/2+A_{ij}/2}$ with $k$ symmetric and $A$ antisymmetric. If $A=0$, then $\pi_i\propto e^{-\mu_i}$.

**Statement.**
(i) $\bar\pi$ is stationary for $\bar Q$.
(ii) $\bar J_{ab}:=\bar\pi_a\bar Q_{ab}-\bar\pi_b\bar Q_{ba}=\sum_{i\in a,j\in b}J_{ij}$, so $\bar J=PJ$. Away from stationarity the same identity holds for $p$ exactly when $p=\bar p\,\Pi$, that is, under local equilibrium inside blocks.
(iii) $\bar\sigma\le\sigma-\sigma_{\rm intra}\le\sigma$ at stationarity, where $\sigma_{\rm intra}\ge0$ is the entropy production of transitions inside blocks.
(iv) (a) For any $\bar\mu$, $\bar Q_{ab}=\bar k_{ab}e^{(\bar\mu_a-\bar\mu_b)/2+\bar A_{ab}/2}$ with $\bar k_{ab}=\sqrt{\bar Q_{ab}\bar Q_{ba}}$ and $\bar A_{ab}=\log(\bar Q_{ab}/\bar Q_{ba})-(\bar\mu_a-\bar\mu_b)$. The coarse cycle affinities $\langle\bar z,\bar A\rangle$ do not depend on $\bar\mu$. (b) If $A=0$, then with $\bar\mu_a=-\log\sum_{i\in a}e^{-\mu_i}$ we get $\bar A=0$ and
$$\bar k_{ab}=\sum_{i\in a,\,j\in b}k_{ij}\sqrt{\pi_{i|a}\,\pi_{j|b}} .$$
(c) If $A\neq0$, the log-sum-exp $\bar\mu$ is only a gauge choice. A cycle-space representative of $\bar A$ requires the projection of Theorem 1(d) on the coarse graph.
(vi) Coarse graining along nested partitions composes, $\mathrm{RG}_{\mathcal B}\circ\mathrm{RG}_{\mathcal A}=\mathrm{RG}_{\mathcal B\circ\mathcal A}$.

**Proof.** (i) $\bar\pi\Pi=\pi$, so $\bar\pi\bar Q=\pi QP=0$. (ii) $\bar\pi_a\bar Q_{ab}=\sum_{i\in a,j\in b}\pi_iQ_{ij}$. The transient statement follows by replacing $\pi_{i|a}$ with $p_i/\bar p_a$. (iii) Write $\sigma=\sum_{i\neq j}x_{ij}\log(x_{ij}/x_{ji})$ with $x_{ij}=\pi_iQ_{ij}$. For each ordered block pair $a\neq b$, the log-sum inequality gives $\sum_{i\in a,j\in b}x_{ij}\log\frac{x_{ij}}{x_{ji}}\ge X_{ab}\log\frac{X_{ab}}{X_{ba}}$ with $X_{ab}=\bar\pi_a\bar Q_{ab}$. The pairs inside blocks contribute $\tfrac12\sum(x_{ij}-x_{ji})\log(x_{ij}/x_{ji})\ge0$. (iv) Part (a) is the split of $\log\bar Q$ into symmetric and antisymmetric parts. Around a coarse cycle the $\bar\mu$ terms cancel. For (b), $A=0$ gives $\pi_iQ_{ij}=\pi_jQ_{ji}$, and summing over $i\in a,\,j\in b$ shows that $\bar Q$ is reversible with respect to $\bar\pi$. Then $\log(\bar Q_{ab}/\bar Q_{ba})=\log(\bar\pi_b/\bar\pi_a)=\bar\mu_a-\bar\mu_b$, because $\bar\pi_a\propto\sum_{i\in a}e^{-\mu_i}=e^{-\bar\mu_a}$. Next, $\pi_iQ_{ij}=Z^{-1}k_{ij}e^{-(\mu_i+\mu_j)/2}$ and $\sqrt{\bar\pi_a\bar\pi_b}=Z^{-1}e^{-(\bar\mu_a+\bar\mu_b)/2}$, and dividing gives $\bar k$. (vi) This follows from $\bar\pi\Pi=\pi$ and the nesting of the conditional weights. $\square$

**Lag propagator (v).** Let $\bar T(\tau)=\Pi e^{\tau Q}P$ (in the column convention, $Pe^{\tau L}\Pi$). Then $\bar\pi\bar T(\tau)=\pi e^{\tau Q}P=\bar\pi$, and $\bar\pi_a\bar T_{ab}(\tau)=\Pr_\pi(X_0\in a,X_\tau\in b)$. So the stationary two-time block statistics and the lagged net fluxes are exact, and $\bar T(\tau)=I+\tau\bar Q+O(\tau^2)$. But $\bar T$ need not satisfy Chapman–Kolmogorov, and $\tau^{-1}\log\bar T(\tau)$ need not be a generator.

*Counterexample.* Take a ring $1\to2\to3\to1$ with forward rate $\gamma$ and backward rate $\epsilon$, so $\pi$ is uniform. Use the blocks $\{1\}$ and $\{2,3\}$. At $\epsilon=0$ the displacement is $N\bmod3$ with $N\sim\mathrm{Poisson}(\lambda)$ and $\lambda=\gamma\tau$, and $p_0=\Pr(N\equiv0)=\tfrac13[1+2e^{-3\lambda/2}\cos(\sqrt3\lambda/2)]$. A direct count gives $\operatorname{tr}\bar T=(1+3p_0)/2$. At $\lambda=2\pi/\sqrt3$, $p_0=\tfrac13(1-2e^{-\sqrt3\pi})\approx0.3304$, so $\det\bar T=\operatorname{tr}\bar T-1\approx-0.0043<0$. A real $2\times2$ matrix with one negative eigenvalue has no real logarithm (Culver 1966), so no generator gives $\bar T(\tau)$. The strict inequality persists for small $\epsilon>0$. The hidden circulation inside $\{2,3\}$ is the memory that a Markov generator cannot hold. $\square$

**Where Chentsov enters.** (i), (ii), (iv), (v) and (vi) are algebra. (iii) is a data-processing inequality for the KL divergence on transition pairs, the monotonicity of Csiszár, and not Chentsov's theorem. Chentsov's theorem (Čencov 1982, Campbell 1986) is a uniqueness statement. Up to scale, the Fisher–Rao metric is the only Riemannian metric on simplices that is invariant under all congruent Markov embeddings, and the $\alpha$-connections are the only invariant affine connections (Amari and Nagaoka 2000). Our $\Pi$ is such an embedding. Wolfer and Watanabe (arXiv:2203.04575) build the kernel version, in which congruent embeddings are linear right inverses of lumping. The local-equilibrium set $\{\bar p\,\Pi\}$ is both e-flat and m-flat. The map $p\mapsto(pP)\Pi$ is the m-projection onto it, and the Pythagorean relation holds. So the claim "the coarse model carries the same geometry as the fine one, and no other geometry does this for every partition" depends on Chentsov. The formulas for $\bar Q$, $\bar\mu$ and $\bar k$ do not. The edge metric $\ell$ and the Maas metric are not covered by Chentsov's theorem, and we make no uniqueness claim for them. Closure of the exponential family is trivial for the full family on a finite set. The content is the explicit map of natural parameters, namely the block log-partition function $\bar\mu_a$, with the block label as the sufficient statistic of the local-equilibrium family. Subfamilies are not closed. For example, $\mu=U/T$ maps to $\bar\mu_a=-\log\sum_{i\in a}e^{-U_i/T}$, which is not affine in $1/T$. When $A=0$, $\bar Q$ is reversible, so the coarse model is again a gradient flow of $\mathrm{KL}(\cdot\|\bar\pi)$ in the Maas metric of the coarse graph (Maas 2011). That coarse metric is not the pushforward of the fine one.

**Label.** (i)–(iii) are known: Esposito (2012), local-equilibrium projection in Markov state models (Hummer and Szabo 2015), and the log-sum inequality. Statements of this project: the closed form (iv)(b) in HyperME parameters, the gauge statement (iv)(c), the explicit counterexample in (v), and the accounting of what does and does not depend on Chentsov.

**Evidence.** On E4, $\bar\pi$ and $\bar J$ match to $10^{-15}$, and $\sigma$ decreases monotonically on $32\to16\to8$ for 64/64 trajectories. On E4b, the lag propagator lowers the $t=T$ error from 0.30 to 0.028, and about half of the off-diagonal entries of $\log\bar T/\tau$ are negative.

## Proposition 4. Wired extrapolation in the knobs

**Statement.** Assume the following. (W1) $\mu=\log c+U/T+\delta\mu(c)$, where $\delta\mu$ does not depend on the knobs. (W2) $A_h=W_h(F)/T+A_h^{\rm chem}$, where $W_h$ is the line integral of the force $F$ along $h$ and the chemostat part is $-\sum_X S_{Xh}(\log c_X+U_X/T)$. (W3) $k_h=T\,m_h$ with $m_h$ knob-independent (the Einstein relation $D=mT$). Then $\log a_h$ and $\log b_h$ are affine in the natural parameters $(1/T,\,F/T,\,\log c_{\rm chem})$. Changing the knobs, monotonically or not, therefore maps the model by substitution, with $m$ and $\delta\mu$ held fixed. For $T\to T'=rT$:
$$k\to r\,k,\qquad U/T\to (U/T)/r,\qquad W/T\to (W/T)/r,\qquad \log c,\ \log c_X\ \text{unchanged}.$$
This matches the G1 rule "mobility $\times r$, every non-entropic force $\div r$, including $A$", with one correction of terms. What scales is the rate prefactor $k\propto D=mT$, while the Einstein mobility $m$ stays fixed. A drive edit $F\to sF$ gives $A\to sA$ on the force part.

(a) If the true system satisfies (W1)–(W3), the log-rate error of a fitted model is the same at every knob value, because the knob terms are exact. (b) If a knob enters as a normalized feature of a universal approximator, data on a training set $K$ of knob values do not constrain predictions at $\kappa'\notin\bar K$. For any continuous target there is a network in the class that fits the training data to any tolerance and matches the target at $\kappa'$. ReLU networks also extrapolate linearly in the input (Xu et al. 2021). That is linear in the rate, while the physics is linear in the log-rate.

**Proof.** (W1)–(W3) give $\log a_h=\log m_h+\log T+\langle\nu_h^+,\log c+\delta\mu\rangle+T^{-1}\big[\langle\nu_h^+,U\rangle+W_h/2\big]+\tfrac12A_h^{\rm chem}$, and similarly for $b_h$. Every knob appears explicitly. For transport, the drift $k(W-\Delta U)/T=m(W-\Delta U)$ does not depend on $T$, and the diffusion $k=mT$ does. This is the SQRA discretization of $\partial_t\rho=\nabla\cdot(mT\nabla\rho+m\rho(\nabla U-F))$. (b) follows from the Tietze extension theorem and universal approximation on a compact set containing $K\cup\{\kappa'\}$. $\square$

**Caveats.** For reactions with a barrier, Eyring's law $k_h\propto T e^{-\Delta G_h^\ddagger/T}$ needs one more scalar per hyperedge class, which data at two temperatures can identify. Debye–Hückel corrections depend on $T$, so (W1) must then be written with the explicit $T$ dependence. After coarse graining, temperature enters through $\bar\mu_a=-\log\sum_{i\in a}e^{-U_i/T}$, which is not affine in $1/T$. The map stays explicit, but it is no longer a rescaling.

**Label.** Local detailed balance with knobs entering as energies divided by $T$, and the Einstein relation, are known (Seifert 2012, Rao and Esposito 2016). Statement of this project: the exact edit map for the wired model class and the non-identifiability contrast (b), which is elementary.

**Evidence.** On E6, a drive of $\times4$ gives error 0.019 against 0.057 for the black box. On G1, the temperature edit gives $10^{-2}$ against 0.34, and keeping $A$ fixed gives 0.31. On C3b, routing the chemostat through the chemical potential lowers the OOD error from 52% to 12%.

## References

Amari and Nagaoka, *Methods of Information Geometry*, AMS (2000). Andersen, Borgan, Gill and Keiding, *Statistical Models Based on Counting Processes*, Springer (1993). Campbell, Proc. AMS 98, 135 (1986). Čencov, *Statistical Decision Rules and Optimal Inference*, AMS (1982). Culver, Proc. AMS 17, 1146 (1966). Esposito, PRE 85, 041125 (2012). Hummer and Szabo, J. Phys. Chem. B 119, 9029 (2015). Jiang, Qian and Qian, *Mathematical Theory of Nonequilibrium Steady States*, LNM 1833 (2004). Kobayashi, Loutchko, Kamimura and Sughiyama, PRR 4, 033208 (2022). Kolchinsky, Dechant, Yoshimura and Ito, arXiv:2206.14599 (2022). Maas, J. Funct. Anal. 261, 2250 (2011). Murashita, Funo and Ueda, PRE 90, 042110 (2014). Rao and Esposito, PRX 6, 041064 (2016). Schnakenberg, Rev. Mod. Phys. 48, 571 (1976). Seifert, Rep. Prog. Phys. 75, 126001 (2012). Wolfer and Watanabe, arXiv:2106.05669 (2021), and arXiv:2203.04575. Xu et al., ICLR (2021). Yoshimura, Kolchinsky, Dechant and Ito, PRR 5, 013017 (2023).
