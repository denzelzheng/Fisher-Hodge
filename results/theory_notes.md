# HyperME theory notes

These notes state five results that the HyperME experiments already support numerically. Each item gives a statement, a proof or proof sketch, a label that separates known results from statements of this project, and one line of numerical evidence. Theorem 0, placed last, states the design rule behind the wired model. It fixes the force by local detailed balance and learns only the symmetric kinetics, lists what this cut guarantees, and says in what precise sense it is the only cut with these guarantees.

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

## Theorem 0. The thermodynamic cut of a gray-box rate model

**Formulation.** Fix the stoichiometry $S$. Let $\mathfrak G(S)$ be the set of rate laws on $S$ that satisfy local detailed balance. A point of $\mathfrak G(S)$ has two coordinates. The condition $\theta$ fixes every force $f_h$ as a natural parameter, and the symmetric kinetic coordinate $k\in(0,\infty)^m$ belongs to the material and fixes the Onsager metric $\Lambda$. A model is a learned map $\xi\mapsto k_w(\xi)$ from local material descriptors to the kinetic coordinate, with weights $w$, joined to the wired map $\theta\mapsto f(\theta)$. For every $w$ and every condition the pair is a point of $\mathfrak G(S)$. Parts (iii)(a) to (d) are properties of $\mathfrak G(S)$ that every such model inherits. Part (iv) shows that models which leave $\mathfrak G(S)$, or its product structure, do not inherit them.

**Setting.** Notation is that of §0, with potentials in units of $T$. A condition is $\theta=(T,F,\mu_Y,U)$, where $\mu_Y$ are the chemical potentials of the chemostatted species, which are coordinates of $\mu$ fixed by $\theta$. The physical potential and force are
$$\mu(c;\theta)=\log c+U/T+\delta\mu(c),\qquad f_h^{\rm phys}(c;\theta)=-\langle S_h,\mu(c;\theta)\rangle+A_h(\theta),\qquad A_h(\theta)=W_h(F)/T,$$
with $W_h(F)$ the line integral of $F$ along $h$. In parts (i) to (iv), $\delta\mu$ is known, possibly zero. A gray-box model sets $(a_h,b_h)=\Phi_h(\lambda_h,c,\theta)$ with $\Phi_h$ fixed and $\lambda_h=K(\xi_h;w)$ learned. In parts (i) to (iv), $\lambda_h$ depends on neither $c$ nor $\theta$. For any model, $\sigma_{\rm phys}(c;\theta)=\sum_hJ_hf_h^{\rm phys}(c;\theta)$ is the entropy production that the physical forces assign to the model's currents.

### (i) Coordinates

**Statement.** The map $(a,b)\mapsto(f,g)=(\log(a/b),\sqrt{ab})$ is a smooth bijection of $(0,\infty)^2$ onto $\mathbb R\times(0,\infty)$ with inverse $a=g\,e^{f/2}$, $b=g\,e^{-f/2}$. Moreover
$$J=2g\sinh(f/2),\qquad \Lambda(a,b)=g\,\frac{\sinh(f/2)}{f/2}\ \ge g,$$
with $\Lambda=g$ at $f=0$. For the rates of §0, $g_h=k_h\varepsilon_h$ with the wired factor $\varepsilon_h=e^{\langle\nu_h^++\nu_h^-,\mu\rangle/2}$, and
$$J_h=k_h\varphi_h,\qquad \varphi_h=2\varepsilon_h\sinh(f_h/2)=e^{\langle\nu_h^+,\mu\rangle+A_h/2}-e^{\langle\nu_h^-,\mu\rangle-A_h/2}.$$

**Proof.** Substitution shows that the two maps are smooth and mutually inverse. Then $a-b=g(e^{f/2}-e^{-f/2})$ and $\Lambda=(a-b)/f$. The inequality is $\sinh x\ge x$ for $x\ge0$ together with evenness. For the §0 rates $\sqrt{a_hb_h}=k_h\varepsilon_h$, and $\varepsilon_he^{\pm f_h/2}$ gives the two exponentials. $\square$

**Label.** Known. Force and symmetric activity are the coordinates of Kaiser, Jack and Zimmer (2018), and $g$ is the frenesy of Maes (2020). $J=2g\sinh(f/2)$ is the kinetic relation of cosh-type gradient systems (Mielke, Peletier and Renger 2014, Peletier and Schlichting 2022).

### (ii) Local detailed balance and the generator set

**Statement.** A rate law satisfies local detailed balance (LDB) at $\theta$ if $\log(a_h/b_h)=f_h^{\rm phys}(c;\theta)$ for all $h$ and $c$. By (i), LDB holds iff $f=f^{\rm phys}$, and then the rates are fixed by $k$ alone. Set
$$\mathfrak G_\theta(S)=\{(a,b):\ f=f^{\rm phys}(\cdot;\theta),\ g_h=k_h\varepsilon_h(\cdot;\theta),\ k\in(0,\infty)^m\},\qquad \mathfrak G(S)=\textstyle\bigcup_{\theta\in\Theta}\mathfrak G_\theta(S),$$
parametrized by $\Theta\times(0,\infty)^m$. The condition enters only through $f$ and $\varepsilon$, linearly in the exponent of the rates (Proposition 4). The thermodynamic cut is "$f$ wired, $k$ learned", and a model in the cut is the map $(\xi,\theta)\mapsto(\theta,k_w(\xi))$.

**Proof.** Apply (i) edge by edge. $\square$

**Label.** LDB with the forces as natural parameters is known (Seifert 2012, Rao and Esposito 2016). The set $\mathfrak G(S)$ as the target of a learned map is the formulation of this project.

### (iii) Properties inherited from $\mathfrak G(S)$

Let the model be in the cut, with any learned output $k\ge0$.

**(a) Second law, equilibria, positivity.** For every $k\ge0$, every $c>0$ and every $\theta$, seen in training or not,
$$J_hf_h^{\rm phys}=2g_hf_h\sinh(f_h/2)\ge0\quad\text{for each }h,$$
with equality iff $f_h=0$ or $k_h=0$. Suppose $[A(\theta')]=0$ and write $A(\theta')=S^\top\psi$. (1) The detailed-balanced states $\{c:S^\top\mu(c;\theta')=A(\theta')\}$ do not depend on $k$. (2) If $\delta\mu=\nabla F_{\rm ex}$ and $F_{\theta'}(c)=\sum_ic_i(\log c_i+U_i/T-1)+F_{\rm ex}(c)$, then $F_{\theta'}(c)-\langle\psi,c\rangle$ is nonincreasing along every solution. (3) At a detailed-balanced state with $H=\nabla^2F_{\theta'}\succ0$, the Jacobian $-S\,\mathrm{diag}(g)\,S^\top H$ has real nonpositive spectrum. (4) A forward step consumes $i$ at a rate with the factor $c_i^{\nu^+_{ih}}$ and a backward step at a rate with $c_i^{\nu^-_{ih}}$, so the positive orthant is invariant when $e^{\delta\mu}$ is bounded on bounded sets. BBKS and modified Patankar schemes keep $c>0$ for every step size (Burchard, Deleersnijder and Meister 2003, Bruggeman et al. 2007). All four statements hold for every $k$, with no penalty term.

**Proof.** The first line is (i) and $x\sinh(x/2)\ge0$. (1) $f^{\rm phys}=0$ is a condition on $c$ only. (2) $\tfrac{d}{dt}F_{\theta'}=\langle\mu,SJ\rangle=\langle A-f^{\rm phys},J\rangle=\tfrac{d}{dt}\langle\psi,c\rangle-\sigma_{\rm phys}$. (3) At $f=0$, $\partial J_h/\partial f_h=g_h$ and the derivative of $g_h$ multiplies $\sinh 0=0$. With $\partial f/\partial c=-S^\top H$ the Jacobian is $M=-SGS^\top H$, and $H^{1/2}MH^{-1/2}=-H^{1/2}SGS^\top H^{1/2}$ is symmetric and negative semidefinite. (4) is cited. $\square$

**Label.** The inequality and (1) to (3) are known for detailed-balanced networks (Horn and Jackson 1972, Maas 2011, Mielke 2011). This project adds that they hold for every output of the learned map at every condition, because they use $f$ alone.

**Evidence.** On C3eq the wired linearization has 0 complex eigenvalues, against 100/128 for CRNN and about 120/128 for the black box. On C5 the $\sigma$ error is 0.008 against 0.36 for CRNN.

**(b) Exact transfer across conditions.** Suppose the truth lies in $\mathfrak G(S)$ with $k^{\rm true}$ a function of $\xi$ only, or with a known temperature law such as $k=mT$ wired into the map. Then for every $c$ and $\theta$,
$$J_h-J_h^{\rm true}=(k_h-k_h^{\rm true})\,\varphi_h(c;\theta),$$
so the relative error of every edge current equals $|k_h-k_h^{\rm true}|/k_h^{\rm true}$ at every condition. With $\delta_k=\max_h|k_h-k_h^{\rm true}|/k_h^{\rm true}$,
$$\|\dot c-\dot c^{\rm true}\|_2\le\|S\|_2\,\|(k-k^{\rm true})\odot\varphi(c;\theta')\|_2\le\|S\|_2\,\delta_k\,\|J^{\rm true}(c;\theta')\|_2 .$$
The new condition enters only through $|\varphi_h|=2\varepsilon_h|\sinh(f_h/2)|$, which grows like $\varepsilon_he^{|f_h|/2}$. The absolute error grows with the drive and the relative error does not. This bounds the vector field, not trajectories. Gronwall gives $\|c(t)-c^{\rm true}(t)\|\le\|S\|_2\delta_k\int_0^te^{L_{\theta'}(t-s)}\|J^{\rm true}(s)\|\,ds$, where the Lipschitz constant $L_{\theta'}$ depends on $\theta'$ and is large near a bifurcation.

**Proof.** Both rate laws lie in $\mathfrak G_\theta(S)$, so they share $f$ and $\varepsilon$ and differ only in $k$. Apply $S$ and the operator norm. The last step is Gronwall's lemma. $\square$

**Label.** This project, elementary. Proposition 4(a) is the per-knob special case.

**Evidence.** E6, drive $\times4$: 0.019 against 0.057. G1, temperature edit: $10^{-2}$ against 0.34. C3, with the chemostat factor $\sqrt b$ in $\varepsilon$, the true constants are exact at every drive strength (0.7% at $10T$).

**(c) Convex estimation.** Take data $(c^i,\dot c^i,\theta^i)_{i\le N}$, the stacked matrix $X=[S\,\mathrm{diag}\,\varphi(c^i;\theta^i)]_i$ and the stacked vector $y$ of the $\dot c^i$. For one $k_h$ per edge, or $k=\Psi\omega$ with fixed features $\Psi$, the equation-error loss $L(k)=\|Xk-y\|^2$ is a convex quadratic. On $k\ge0$ it is a non-negative least-squares problem. A global minimizer $k^*$ exists, every local minimizer is global, $Xk^*$ is unique, and $k^*$ is unique if $\ker X=0$. For $k=\exp N_w(\xi)$ with a network $N_w$, $L$ is in general not convex in $w$. Still $L(k^*)\le L(\exp N_w(\xi))$ for every $w$, and a network that fits $\log k^*$ on the observed descriptors reaches the global optimum. This is the closed-form step followed by distillation. Convexity holds for the equation-error loss, not for the rollout loss.

**Proof.** $L$ is a squared norm of an affine map. A quadratic bounded below attains its minimum on a polyhedron (Frank and Wolfe 1956). Strict convexity of $\|\cdot\|^2$ gives uniqueness of $Xk^*$. The bound holds because $k^*$ minimizes over a larger set. If some $k^*_h=0$, the bound is approached and not attained. $\square$

**Label.** Elementary, the structure of SINDy (Brunton, Proctor and Kutz 2016) and of NNLS (Lawson and Hanson 1974). This project: the cut puts the gray-box problem in this form.

**Evidence.** C3, OOD error 12.3% to 0.30% after the closed-form step. C5, 0.25% in and 3.6% out of distribution, and the minimum that switched the reactions off is gone.

**(d) Identifiability.** Noise-free data determine $k$ iff $\ker X=0$, and $v\in\ker X$ iff $\mathrm{diag}(\varphi^i)v\in\ker S$ for all $i$. The unresolved directions of $k$ are those that the data map into the blind spot $\ker S$ of Theorem 1(a).

*Lemma (parallel channels).* If $S_h=\alpha S_{h'}$ with $\alpha\neq0$ and $\varphi_h=\beta\varphi_{h'}$ at every data point, then $e_h-\alpha\beta\,e_{h'}\in\ker X$, and $k_h,k_{h'}$ are not separately identified.

*Example.* For the C1 pair $X\to X_p$ and $X_p\to X$, $S_2=-S_1$. With $r=y/x$, the ratio $\varphi_2/\varphi_1=(e^{A_2/2}r-e^{-A_2/2})/(e^{A_1/2}-e^{-A_1/2}r)$ is a Möbius map of determinant $2\sinh(C/2)$, $C=A_1+A_2$. If $C\neq0$ and the data contain two values of $r$, then $k_1$ and $k_2$ are identified. If $C=0$, then $\varphi_2=-\varphi_1$ and only $k_1+k_2$ is identified. In Proposition 1′ the same data cannot fix $C$ when $A$ is free. Wiring the force moves the cycle affinity from the data into the model, and the kinetic problem that remains is identifiable. If both channels are nearly irreversible, $\varphi_1$ and $\varphi_2$ are proportional up to reverse terms, and the smallest singular value of $X$ is of that order. This is the deterministic counterpart of Theorem 2.

**Proof.** $Xv=0$ iff $S\,\mathrm{diag}(\varphi^i)v=0$ for all $i$. For the lemma, $S(\varphi_he_h-\alpha\beta\varphi_{h'}e_{h'})=\alpha\beta\varphi_{h'}S_{h'}-\alpha\beta\varphi_{h'}S_{h'}=0$. In the example, $v\in\ker X$ gives $v_1\varphi_1^i=v_2\varphi_2^i$ for all $i$. A Möbius map with nonzero determinant is injective, so two values of $r$ force $v=0$. $\square$

**Label.** The criterion is elementary. The lemma, the Möbius criterion and the contrast with Proposition 1′ are of this project.

**Evidence.** On C3 the entropy production is off by 22–28% because two channels remove $X$ at rates proportional to $x$. One reservoir observable separates them and raises the lower bound from 0.58 to 0.80 of the true value.

### (iv) Necessity: models that leave $\mathfrak G(S)$

**Proposition 0.1 (uniqueness at the flux level).** Fix $h$. Let a model give $J_h=\Psi_h(\lambda,c,\theta)$ with $\lambda\in\mathcal L\subset\mathbb R^p$ independent of $\theta$. Let the truth be in $\mathfrak G(S)$ with $k_h^{\rm true}=\kappa>0$ independent of $\theta$, so $J_h^{\rm true}=\kappa\varphi_h$. Assume transfer: (T) for all $\kappa>0$ and $\lambda\in\mathcal L$, if $\Psi_h(\lambda,\cdot,\theta)=\kappa\varphi_h(\cdot;\theta)$ for all $\theta$ in the training set $\Theta_0$, then the same holds for all $\theta\in\Theta$. Assume $\varphi_h\not\equiv0$ on $\Theta_0$, and let $\mathcal L_R$ be the set of $\lambda$ that fit some $\kappa$ on $\Theta_0$.
(1) There is $\rho:\mathcal L_R\to(0,\infty)$ with $\Psi_h(\lambda,c,\theta)=\rho(\lambda)\varphi_h(c;\theta)$ for all $c$ and all $\theta\in\Theta$. Every fitted model is a point of $\mathfrak G(S)$ at every condition.
(2) If $\Psi_h$ is affine in $\lambda$ and $\mathcal L_R$ affinely spans $\mathbb R^p$, then $\Psi_h=(\alpha_0+\langle\alpha,\lambda\rangle)\varphi_h$ for every $\lambda$. This is the thermodynamic cut with $k_h$ affine in $\lambda$.

**Proof.** (1) For $\lambda\in\mathcal L_R$ pick $\kappa$ and extend by (T). $\kappa$ is unique because $\kappa\varphi_h=\kappa'\varphi_h$ with $\varphi_h\not\equiv0$ forces $\kappa=\kappa'$. Set $\rho(\lambda)=\kappa$. (2) Write $\Psi_h=\psi_0+\langle\lambda,\psi\rangle$ and take affinely independent $\lambda^0,\dots,\lambda^p\in\mathcal L_R$. Then $\langle\lambda^j-\lambda^0,\psi\rangle=(\rho(\lambda^j)-\rho(\lambda^0))\varphi_h$, and since the $\lambda^j-\lambda^0$ form a basis, each $\psi_l=\alpha_l\varphi_h$. Then $\psi_0=\rho(\lambda^0)\varphi_h-\langle\lambda^0,\psi\rangle=\alpha_0\varphi_h$. $\square$

**What "only" means.** Among gray-box models whose learned part does not see $\theta$, exact transfer (T) forces every fitted model onto $\mathfrak G(S)$ at the level of currents. Property (a) then follows, and if the learned part enters linearly, as (c) requires, $k$ is fixed up to an affine map. Three limits apply. Property (a) alone forces only $\mathrm{sign}\,f=\mathrm{sign}\,f^{\rm phys}$. Property (c) alone is shared by a mass-action CRNN with free forward and backward constants, which fails (a) and (T) at a new $\theta$. Proposition 0.1 fixes $J$, not $(a,b)$. For every $C>0$ the pair $\tilde f=2\,\mathrm{arcsinh}(C\sinh(f^{\rm phys}/2))$, $\tilde g=g/C$ gives the same $J$, so it is invisible to deterministic data and to $\sigma_{\rm phys}$. LDB at the level of rates is a modelling axiom. Jump-level data can test it (Theorem 2).

**Counterexamples.** All three use one loop. Nodes 1 and 2 carry $x=c_1$, $y=c_2$, and edges $e_1,e_2$ both run from 1 to 2 with $W_1=w$, $W_2=-w$ and $T=1$, $U=0$. Then $f_1-f_2=2w$, $J_1=k_1(xe^{w/2}-ye^{-w/2})$, $J_2=k_2(xe^{-w/2}-ye^{w/2})$, and $\dot y=J_1+J_2$. Training uses $w\in\Theta_0=[0,w_0]$, and the test uses $w'>w_0$.

(a′) *UDE closure,* $J_i=N(c,\xi_i,\theta)$. For any continuous $z(c,\theta)$ the closure $(J_1^{\rm true}-z,\ J_2^{\rm true}+z)$ gives the same $\dot c$ at every condition, and $\sigma_{\rm phys}=\sigma^{\rm true}-2wz$. Take $z=0$ on $\Theta_0$ and $z>\sigma^{\rm true}/(2w')$ at $w'$. The closure fits the training data exactly, and $\sigma_{\rm phys}<0$ at $w'$. The violation is a circulating current in $\ker S$, so it stays invisible in $\dot c$ even at $w'$. A universal approximator matches this closure on a compact set to any tolerance, and the strict inequality survives.

(b′) *Condition as a feature of $k$.* Keep $f$ wired and let $k_1=k_1^{\rm true}+\eta(\theta)$ with $\eta=0$ on $\Theta_0$ and $\eta(w')=M>-k_1^{\rm true}$. The model fits the training data and stays in each fibre $\mathfrak G_\theta(S)$, so (a) holds. But $k$ now moves with $\theta$, which breaks the product structure of $\mathfrak G(S)$. The relative error of $J_1$ at $w'$ is $M/k_1^{\rm true}$, which is arbitrary. If $\theta$ is a feature of a learned force, $f=f^{\rm phys}+\eta(\theta)$, the model leaves $\mathfrak G(S)$ and (a) fails as well.

(c′) *Learned force coefficient.* Fix $k_1=1+\delta$, $k_2=1-\delta$ and learn $u$ in $A_i=uW_i$, a learned field strength or harmonic drive coefficient. Then $\dot y(u)=2(x-y)\cosh(uw/2)+2\delta(x+y)\sinh(uw/2)$. Let the data come from $u^*\neq0$ and let $L_\delta(u)=\sum_i(\dot y^i(u)-\dot y^i(u^*))^2$. At $\delta=0$, $L_0$ is even, vanishes at $\pm u^*$ and is positive at $0$ if some $x_i\neq y_i$, so it is not convex. Also $L_0''(\pm u^*)=2w^2\sinh^2(u^*w/2)\sum_i(x_i-y_i)^2>0$. By the implicit function theorem, for small $\delta$ there is a strict local minimizer near $-u^*$. If the vectors $(x_i-y_i)_i$ and $(x_i+y_i)_i$ are not parallel, all residuals vanish only at $u=u^*$, so this minimizer has $L_\delta>0$ and is spurious. In the cut, $u=1$ is wired and $(k_1,k_2)$ enter linearly. They are identified when $w\neq0$, because the Möbius determinant of $\varphi_2/\varphi_1$ is $-2\sinh w$.

**Label.** The constructions are elementary, and (b′) restates Proposition 4(b). Proposition 0.1 and the reading of "only" are of this project.

**Evidence.** G1 with $A$ held fixed: 0.31. C3 with normalized chemostat features: 52–56%, against 12.3% through the potential channel. On C2 a nonconvex learned free energy created a spurious bistability. The H1 UDE-closure ablation and the count of edges with $J_hf_h^{\rm phys}<0$ are pending.

### (v) Boundaries

*Concentration-dependent $k$* (crowding). $\mathfrak G(S)$ becomes infinite-dimensional, with $k_h$ a positive function of $c$. (a) is unchanged, since it uses only $g\ge0$ at the current state. (c) holds for $k=\sum_j\omega_j\psi_j(\xi,c)$ with fixed features. In (b) the identity $J-J^{\rm true}=(k-k^{\rm true})\varphi$ still holds pointwise. At a new condition the solution may visit concentrations outside the training range, and the error of $k$ there is extrapolation in $c$, which the theorem does not cover.

*Temperature.* A known law such as $k=mT$ is wired. Eyring's $k=mTe^{-\Delta G^\ddagger/T}$ adds one scalar per class. NNLS at each temperature, followed by linear regression of $\log(k/T)$ on $1/T$, keeps both steps convex.

*Learned $\delta\mu$.* The force is computed from the learned potential, so the model lies in the generator set of the learned free energy. $\sum_hJ_hf_h^{\rm model}\ge0$ for every learned $\delta\mu$, and (a)(2) holds for the learned $F$. $\sigma_{\rm phys}$ is controlled only as far as $\delta\mu$ is right. (b) needs $\delta\mu$ independent of $\theta$ (W1) and correct at the concentrations reached at $\theta'$. (c) is convex in $k$ for fixed $\delta\mu$, not jointly. On C5 the learned $\ln\gamma$ is flat outside the training range (0.9 against 1.17). Learned harmonic drive coefficients are part of $f$, so (c′) applies to them.

### Remark: earlier results as structure of $\mathfrak G(S)$

The blind spot (Theorem 1) says that density data are constant along cycle directions of the set of rate laws with free $[A]$, and Proposition 1′ exhibits one such fibre. Within $\mathfrak G(S)$, where $[A]$ is wired, the fibre reduces to $\ker X$ of (iii)(d). Closed-form coarse graining (Theorem 3) maps $\mathfrak G(S)$ at $A=0$ into $\mathfrak G(\bar S)$, with the explicit $\bar k$ and the log-sum-exp $\bar\mu$, and the stationary statistics are exact. For $A\neq0$ the coarse generator again has the coordinates of (i). Its cycle affinities are fixed, and its potential part is a gauge choice. In minimum-dissipation control (WS5) the cost is $\int\sum_hJ_h^2/\Lambda_h\,dt$, the metric that $k$ fixes on $\mathfrak G(S)$. The wired protocols reach 1.01 times the Benamou–Brenier value of a convex program, against 1.09 for the black box. We claim no uncertainty relation here, only this convex lower bound.

**Relation to known theory.** The following structures are known: the exponential family of rates under LDB (Seifert 2012, Rao and Esposito 2016), cycle affinities (Schnakenberg 1976), the Hessian geometry of reaction networks (Kobayashi et al. 2022), and the gradient structure of detailed-balanced dynamics (Maas 2011, Mielke 2011). Near equilibrium $\Lambda\to g$, so $g$ is the edge Onsager coefficient and reciprocity holds by construction. At $[A]=0$, $\dot c=-S\Lambda S^\top\mu$ is the gradient flow of $F$ in the metric of $\Lambda$, so the cut wires the energy and learns the dissipation. The contribution of this project is the learning formulation, a map from material descriptors into $\mathfrak G(S)$, together with the combination of (a) to (d) that it yields at once, Proposition 0.1, and the counterexamples. Each step is elementary.

## References

Amari and Nagaoka, *Methods of Information Geometry*, AMS (2000). Andersen, Borgan, Gill and Keiding, *Statistical Models Based on Counting Processes*, Springer (1993). Brunton, Proctor and Kutz, PNAS 113, 3932 (2016). Bruggeman, Burchard, Kooi and Sommeijer, Appl. Numer. Math. 57, 36 (2007). Burchard, Deleersnijder and Meister, Appl. Numer. Math. 47, 1 (2003). Campbell, Proc. AMS 98, 135 (1986). Čencov, *Statistical Decision Rules and Optimal Inference*, AMS (1982). Culver, Proc. AMS 17, 1146 (1966). Esposito, PRE 85, 041125 (2012). Frank and Wolfe, Naval Res. Logist. Q. 3, 95 (1956). Horn and Jackson, Arch. Ration. Mech. Anal. 47, 81 (1972). Hummer and Szabo, J. Phys. Chem. B 119, 9029 (2015). Jiang, Qian and Qian, *Mathematical Theory of Nonequilibrium Steady States*, LNM 1833 (2004). Kaiser, Jack and Zimmer, J. Stat. Phys. 170, 1019 (2018). Kobayashi, Loutchko, Kamimura and Sughiyama, PRR 4, 033208 (2022). Kolchinsky, Dechant, Yoshimura and Ito, arXiv:2206.14599 (2022). Lawson and Hanson, *Solving Least Squares Problems*, Prentice-Hall (1974). Maas, J. Funct. Anal. 261, 2250 (2011). Maes, Phys. Rep. 850, 1 (2020). Mielke, Nonlinearity 24, 1329 (2011). Mielke, Peletier and Renger, Potential Anal. 41, 1293 (2014), arXiv:1312.7591. Murashita, Funo and Ueda, PRE 90, 042110 (2014). Peletier and Schlichting, arXiv:2203.05435 (2022). Rao and Esposito, PRX 6, 041064 (2016). Schnakenberg, Rev. Mod. Phys. 48, 571 (1976). Seifert, Rep. Prog. Phys. 75, 126001 (2012). Wolfer and Watanabe, arXiv:2106.05669 (2021), and arXiv:2203.04575. Xu et al., ICLR (2021). Yoshimura, Kolchinsky, Dechant and Ito, PRR 5, 013017 (2023).
