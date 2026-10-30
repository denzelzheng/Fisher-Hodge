# Theory of the Neural Chart

The Neural Chart is a master equation on graphs and hypergraphs whose thermodynamic force is fixed by physics and whose mobility is learned. This document states the theory behind it: the setting, the design rule (Theorem 0) and its guarantees, results on observability, estimation limits, coarse graining and extrapolation, and a restatement of these results in information geometry. Each result has a statement, a proof or proof sketch, and a status line that separates known results from new ones.

**Contents**

- 0. Setting and notation (gauge lemma, $\beta_1$)
- Theorem 0. The thermodynamic cut of a gray-box rate model
  - (i) Coordinates
  - (ii) Local detailed balance and the generator set
  - (iii) Properties inherited from $\mathfrak G(S)$: (a) second law, equilibria, positivity, (b) exact transfer across conditions, (c) convex estimation, (d) identifiability
  - (iv) Necessity: models that leave $\mathfrak G(S)$
  - (v) Boundaries
  - Remark: the other results as structure of $\mathfrak G(S)$
- Theorem 1. Minimal observables and the blind spot of densities
- Theorem 2. Cramér–Rao bound for the affinity of a near-irreversible hyperedge
- Theorem 3. Closed-form coarse graining
- Proposition 4. Extrapolation in the conditions
- Information geometry of the chart (IG1 to IG7, and the division of labour between information geometry and Hodge theory)
- References

## 0. Setting and notation

There are $n$ state coordinates (species at sites) and $m$ hyperedges. Hyperedge $h$ has reactant and product vectors $\nu_h^+,\nu_h^-\in\mathbb{Z}_{\ge 0}^n$, and $S=\nu^--\nu^+\in\mathbb{R}^{n\times m}$. The rates are

$$a_h=k_h\,e^{\langle\nu_h^+,\mu\rangle+A_h/2},\qquad b_h=k_h\,e^{\langle\nu_h^-,\mu\rangle-A_h/2},\qquad J=a-b,\qquad \dot c=SJ,$$

with $k_h>0$ and $\mu=\partial F/\partial c$. The edge force is $f_h=\log(a_h/b_h)=-\langle S_h,\mu\rangle+A_h$. With the logarithmic mean $\ell_h=\Lambda(a_h,b_h)=(a_h-b_h)/(\log a_h-\log b_h)$ the Onsager relation $J=\ell\odot f$ holds exactly, and $\sigma=\sum_h J_hf_h=\sum_h\ell_hf_h^2\ge 0$.

**Gauge lemma.** For any $\varphi\in\mathbb{R}^n$ the map $(\mu,A,k_h)\mapsto(\mu+\varphi,\ A+S^\top\varphi,\ k_h e^{-\langle\nu_h^++\nu_h^-,\varphi\rangle/2})$ leaves every $a_h$ and $b_h$ unchanged. *Proof.* Both exponents gain $\langle\nu_h^++\nu_h^-,\varphi\rangle/2$, which the new $k_h$ cancels. $\square$ So only the class $[A]\in\mathbb{R}^m/\operatorname{im}S^\top$ is meaningful unless the potential part of $\mu$ is fixed by other means.

We write $\beta_1:=\dim\ker S=m-\operatorname{rank}S$. For a graph incidence matrix $\operatorname{rank}S=|V|-c$, so $\beta_1=|E|-|V|+c$, the first Betti number. For a hypergraph we use $\beta_1$ as the definition, through the two-term chain complex $\mathbb{R}^m\xrightarrow{S}\mathbb{R}^n$ with $H_1=\ker S$ and $H_0=\operatorname{coker}S$. The dimension of $H_0$ counts the conservation laws.

## Theorem 0. The thermodynamic cut of a gray-box rate model

**Formulation.** Fix the stoichiometry $S$. Let $\mathfrak G(S)$ be the set of rate laws on $S$ that satisfy local detailed balance. A point of $\mathfrak G(S)$ has two coordinates. The condition $\theta$ fixes every force $f_h$ as a natural parameter. The symmetric kinetic coordinate $k\in(0,\infty)^m$ belongs to the material and fixes the Onsager metric $\Lambda$. A model is a learned map $\xi\mapsto k_w(\xi)$ from local material descriptors to the kinetic coordinate, with weights $w$, joined to the map $\theta\mapsto f(\theta)$ that physics fixes. Such a model is a Neural Chart, defined formally in the information-geometry section. For every $w$ and every condition the pair is a point of $\mathfrak G(S)$. Parts (iii)(a) to (d) are properties of $\mathfrak G(S)$ that every such model inherits. Part (iv) shows that models which leave $\mathfrak G(S)$, or its product structure, do not inherit them.

**Setting.** Notation is that of §0, with potentials in units of $T$. A condition is $\theta=(T,F,\mu_Y,U)$, where $\mu_Y$ are the chemical potentials of the chemostatted species, which are coordinates of $\mu$ fixed by $\theta$. The physical potential and force are

$$\mu(c;\theta)=\log c+U/T+\delta\mu(c),\qquad f_h^{\rm phys}(c;\theta)=-\langle S_h,\mu(c;\theta)\rangle+A_h(\theta),\qquad A_h(\theta)=W_h(F)/T,$$

with $W_h(F)$ the line integral of $F$ along $h$. In parts (i) to (iv), $\delta\mu$ is known, possibly zero. A gray-box model sets $(a_h,b_h)=\Phi_h(\lambda_h,c,\theta)$ with $\Phi_h$ fixed and $\lambda_h=K(\xi_h;w)$ learned. In parts (i) to (iv), $\lambda_h$ depends on neither $c$ nor $\theta$. For any model, $\sigma_{\rm phys}(c;\theta)=\sum_hJ_hf_h^{\rm phys}(c;\theta)$ is the entropy production that the physical forces assign to the model's currents.

### (i) Coordinates

**Statement.** The map $(a,b)\mapsto(f,g)=(\log(a/b),\sqrt{ab})$ is a smooth bijection of $(0,\infty)^2$ onto $\mathbb R\times(0,\infty)$ with inverse $a=g\,e^{f/2}$, $b=g\,e^{-f/2}$. Moreover

$$J=2g\sinh(f/2),\qquad \Lambda(a,b)=g\,\frac{\sinh(f/2)}{f/2}\ \ge g,$$

with $\Lambda=g$ at $f=0$. For the rates of §0, $g_h=k_h\varepsilon_h$ with the factor $\varepsilon_h=e^{\langle\nu_h^++\nu_h^-,\mu\rangle/2}$, which physics fixes, and

$$J_h=k_h\varphi_h,\qquad \varphi_h=2\varepsilon_h\sinh(f_h/2)=e^{\langle\nu_h^+,\mu\rangle+A_h/2}-e^{\langle\nu_h^-,\mu\rangle-A_h/2}.$$

**Proof.** Substitution shows that the two maps are smooth and mutually inverse. Then $a-b=g(e^{f/2}-e^{-f/2})$ and $\Lambda=(a-b)/f$. The inequality is $\sinh x\ge x$ for $x\ge0$ together with evenness. For the §0 rates $\sqrt{a_hb_h}=k_h\varepsilon_h$, and $\varepsilon_he^{\pm f_h/2}$ gives the two exponentials. $\square$

**Status.** Known. Force and symmetric activity are the coordinates of Kaiser, Jack and Zimmer (2018), and $g$ is the frenesy of Maes (2020). The relation $J=2g\sinh(f/2)$ is the kinetic relation of cosh-type gradient systems (Mielke, Peletier and Renger 2014, Peletier and Schlichting 2022).

### (ii) Local detailed balance and the generator set

**Statement.** A rate law satisfies local detailed balance (LDB) at $\theta$ if $\log(a_h/b_h)=f_h^{\rm phys}(c;\theta)$ for all $h$ and $c$. By (i), LDB holds iff $f=f^{\rm phys}$, and then the rates are fixed by $k$ alone. Set

$$\mathfrak G_\theta(S)=\{(a,b):\ f=f^{\rm phys}(\cdot;\theta),\ g_h=k_h\varepsilon_h(\cdot;\theta),\ k\in(0,\infty)^m\},\qquad \mathfrak G(S)=\textstyle\bigcup_{\theta\in\Theta}\mathfrak G_\theta(S),$$

parametrized by $\Theta\times(0,\infty)^m$. The condition enters only through $f$ and $\varepsilon$, linearly in the exponent of the rates (Proposition 4). The thermodynamic cut fixes $f$ by physics and learns $k$. A model in the cut is the map $(\xi,\theta)\mapsto(\theta,k_w(\xi))$.

**Proof.** Apply (i) edge by edge. $\square$

**Status.** Known: LDB with the forces as natural parameters (Seifert 2012, Rao and Esposito 2016). New here: the set $\mathfrak G(S)$ as the target of a learned map.

### (iii) Properties inherited from $\mathfrak G(S)$

Let the model be in the cut, with any learned output $k\ge0$.

**(a) Second law, equilibria, positivity.** For every $k\ge0$, every $c>0$ and every $\theta$, seen in training or not,

$$J_hf_h^{\rm phys}=2g_hf_h\sinh(f_h/2)\ge0\quad\text{for each }h,$$

with equality iff $f_h=0$ or $k_h=0$. Suppose $[A(\theta')]=0$ and write $A(\theta')=S^\top\psi$. (1) The detailed-balanced states $\{c:S^\top\mu(c;\theta')=A(\theta')\}$ do not depend on $k$. (2) If $\delta\mu=\nabla F_{\rm ex}$ and $F_{\theta'}(c)=\sum_ic_i(\log c_i+U_i/T-1)+F_{\rm ex}(c)$, then $F_{\theta'}(c)-\langle\psi,c\rangle$ is nonincreasing along every solution. (3) At a detailed-balanced state with $H=\nabla^2F_{\theta'}\succ0$, the Jacobian $-S\,\mathrm{diag}(g)\,S^\top H$ has real nonpositive spectrum. (4) A forward step consumes $i$ at a rate with the factor $c_i^{\nu^+_{ih}}$ and a backward step at a rate with $c_i^{\nu^-_{ih}}$, so the positive orthant is invariant when $e^{\delta\mu}$ is bounded on bounded sets. BBKS and modified Patankar schemes keep $c>0$ for every step size (Burchard, Deleersnijder and Meister 2003, Bruggeman et al. 2007). All four statements hold for every $k$, with no penalty term.

**Proof.** The first line is (i) and $x\sinh(x/2)\ge0$. (1) $f^{\rm phys}=0$ is a condition on $c$ only. (2) $\tfrac{d}{dt}F_{\theta'}=\langle\mu,SJ\rangle=\langle A-f^{\rm phys},J\rangle=\tfrac{d}{dt}\langle\psi,c\rangle-\sigma_{\rm phys}$. (3) At $f=0$, $\partial J_h/\partial f_h=g_h$ and the derivative of $g_h$ multiplies $\sinh 0=0$. With $\partial f/\partial c=-S^\top H$ the Jacobian is $M=-SGS^\top H$, and $H^{1/2}MH^{-1/2}=-H^{1/2}SGS^\top H^{1/2}$ is symmetric and negative semidefinite. (4) is cited. $\square$

**Status.** Known for detailed-balanced networks: the inequality and (1) to (3) (Horn and Jackson 1972, Maas 2011, Mielke 2011). New here: they hold for every output of the learned map at every condition, because they use $f$ alone.

**(b) Exact transfer across conditions.** Suppose the truth lies in $\mathfrak G(S)$ with $k^{\rm true}$ a function of $\xi$ only, or with a known temperature law such as $k=mT$ built into the map. Then for every $c$ and $\theta$,

$$J_h-J_h^{\rm true}=(k_h-k_h^{\rm true})\,\varphi_h(c;\theta),$$

so the relative error of every edge current equals $|k_h-k_h^{\rm true}|/k_h^{\rm true}$ at every condition. With $\delta_k=\max_h|k_h-k_h^{\rm true}|/k_h^{\rm true}$,

$$\|\dot c-\dot c^{\rm true}\|_2\le\|S\|_2\,\|(k-k^{\rm true})\odot\varphi(c;\theta')\|_2\le\|S\|_2\,\delta_k\,\|J^{\rm true}(c;\theta')\|_2 .$$

The new condition enters only through $|\varphi_h|=2\varepsilon_h|\sinh(f_h/2)|$, which grows like $\varepsilon_he^{|f_h|/2}$. The absolute error grows with the drive and the relative error does not. This bounds the vector field. For trajectories, Gronwall gives $\|c(t)-c^{\rm true}(t)\|\le\|S\|_2\delta_k\int_0^te^{L_{\theta'}(t-s)}\|J^{\rm true}(s)\|\,ds$, where the Lipschitz constant $L_{\theta'}$ depends on $\theta'$ and is large near a bifurcation.

**Proof.** Both rate laws lie in $\mathfrak G_\theta(S)$, so they share $f$ and $\varepsilon$ and differ only in $k$. Apply $S$ and the operator norm. The last step is Gronwall's lemma. $\square$

**Status.** New here. Elementary. Proposition 4(a) is a special case.

**(c) Convex estimation.** Take data $(c^i,\dot c^i,\theta^i)_{i\le N}$, the stacked matrix $X=[S\,\mathrm{diag}\,\varphi(c^i;\theta^i)]_i$ and the stacked vector $y$ of the $\dot c^i$. For one $k_h$ per edge, or $k=\Psi\omega$ with fixed features $\Psi$, the equation-error loss $L(k)=\|Xk-y\|^2$ is a convex quadratic. On $k\ge0$ it is a non-negative least-squares (NNLS) problem. A global minimizer $k^*$ exists, every local minimizer is global, $Xk^*$ is unique, and $k^*$ is unique if $\ker X=0$. For $k=\exp N_w(\xi)$ with a network $N_w$, $L$ is in general not convex in $w$. Still $L(k^*)\le L(\exp N_w(\xi))$ for every $w$, and a network that fits $\log k^*$ on the observed descriptors reaches the global optimum. This is the closed-form step followed by distillation. Convexity holds for the equation-error loss. The rollout loss is in general not convex.

**Proof.** $L$ is a squared norm of an affine map. A quadratic bounded below attains its minimum on a polyhedron (Frank and Wolfe 1956). Strict convexity of $\|\cdot\|^2$ gives uniqueness of $Xk^*$. The bound holds because $k^*$ minimizes over a larger set. If some $k^*_h=0$, the bound is approached and not attained. $\square$

**Status.** Elementary. It is the structure of SINDy (Brunton, Proctor and Kutz 2016) and of NNLS (Lawson and Hanson 1974). New here: the cut puts the gray-box problem in this form.

**(d) Identifiability.** Noise-free data determine $k$ iff $\ker X=0$, and $v\in\ker X$ iff $\mathrm{diag}(\varphi^i)v\in\ker S$ for all $i$. The unresolved directions of $k$ are those that the data map into the blind spot $\ker S$ of Theorem 1(a).

*Lemma (parallel channels).* If $S_h=\alpha S_{h'}$ with $\alpha\neq0$ and $\varphi_h=\beta\varphi_{h'}$ at every data point, then $e_h-\alpha\beta\,e_{h'}\in\ker X$, and $k_h,k_{h'}$ are not separately identified.

*Example (the driven phosphorylation cycle).* For the pair $X\to X_p$ and $X_p\to X$, $S_2=-S_1$. With $r=y/x$, the ratio $\varphi_2/\varphi_1=(e^{A_2/2}r-e^{-A_2/2})/(e^{A_1/2}-e^{-A_1/2}r)$ is a Möbius map of determinant $2\sinh(C/2)$, $C=A_1+A_2$. If $C\neq0$ and the data contain two values of $r$, then $k_1$ and $k_2$ are identified. If $C=0$, then $\varphi_2=-\varphi_1$ and only $k_1+k_2$ is identified. In Proposition 1′ the same data cannot fix $C$ when $A$ is free. Fixing the force by physics moves the cycle affinity from the data into the model, and the kinetic problem that remains is identifiable. If both channels are nearly irreversible, $\varphi_1$ and $\varphi_2$ are proportional up to reverse terms, and the smallest singular value of $X$ is of that order. This is the deterministic counterpart of Theorem 2.

**Proof.** $Xv=0$ iff $S\,\mathrm{diag}(\varphi^i)v=0$ for all $i$. For the lemma, $S(\varphi_he_h-\alpha\beta\varphi_{h'}e_{h'})=\alpha\beta\varphi_{h'}S_{h'}-\alpha\beta\varphi_{h'}S_{h'}=0$. In the example, $v\in\ker X$ gives $v_1\varphi_1^i=v_2\varphi_2^i$ for all $i$. A Möbius map with nonzero determinant is injective, so two values of $r$ force $v=0$. $\square$

**Status.** The criterion is elementary. New here: the lemma, the Möbius criterion and the contrast with Proposition 1′.

### (iv) Necessity: models that leave $\mathfrak G(S)$

**Proposition 0.1 (uniqueness at the flux level).** Fix $h$. Let a model give $J_h=\Psi_h(\lambda,c,\theta)$ with $\lambda\in\mathcal L\subset\mathbb R^p$ independent of $\theta$. Let the truth be in $\mathfrak G(S)$ with $k_h^{\rm true}=\kappa>0$ independent of $\theta$, so $J_h^{\rm true}=\kappa\varphi_h$. Assume the following transfer property. (T) For all $\kappa>0$ and $\lambda\in\mathcal L$, if $\Psi_h(\lambda,\cdot,\theta)=\kappa\varphi_h(\cdot;\theta)$ for all $\theta$ in the training set $\Theta_0$, then the same holds for all $\theta\in\Theta$. Assume $\varphi_h\not\equiv0$ on $\Theta_0$, and let $\mathcal L_R$ be the set of $\lambda$ that fit some $\kappa$ on $\Theta_0$.

(1) There is $\rho:\mathcal L_R\to(0,\infty)$ with $\Psi_h(\lambda,c,\theta)=\rho(\lambda)\varphi_h(c;\theta)$ for all $c$ and all $\theta\in\Theta$. Every fitted model is a point of $\mathfrak G(S)$ at every condition.

(2) If $\Psi_h$ is affine in $\lambda$ and $\mathcal L_R$ affinely spans $\mathbb R^p$, then $\Psi_h=(\alpha_0+\langle\alpha,\lambda\rangle)\varphi_h$ for every $\lambda$. This is the thermodynamic cut with $k_h$ affine in $\lambda$.

**Proof.** (1) For $\lambda\in\mathcal L_R$ pick $\kappa$ and extend by (T). $\kappa$ is unique because $\kappa\varphi_h=\kappa'\varphi_h$ with $\varphi_h\not\equiv0$ forces $\kappa=\kappa'$. Set $\rho(\lambda)=\kappa$. (2) Write $\Psi_h=\psi_0+\langle\lambda,\psi\rangle$ and take affinely independent $\lambda^0,\dots,\lambda^p\in\mathcal L_R$. Then $\langle\lambda^j-\lambda^0,\psi\rangle=(\rho(\lambda^j)-\rho(\lambda^0))\varphi_h$, and since the $\lambda^j-\lambda^0$ form a basis, each $\psi_l=\alpha_l\varphi_h$. Then $\psi_0=\rho(\lambda^0)\varphi_h-\langle\lambda^0,\psi\rangle=\alpha_0\varphi_h$. $\square$

**What "only" means.** Among gray-box models whose learned part does not see $\theta$, exact transfer (T) forces every fitted model onto $\mathfrak G(S)$ at the level of currents. Property (a) then follows. If the learned part enters linearly, as (c) requires, $k$ is fixed up to an affine map. Three limits apply. Property (a) alone forces only $\mathrm{sign}\,f=\mathrm{sign}\,f^{\rm phys}$. Property (c) alone is shared by a mass-action chemical reaction neural network (CRNN) with free forward and backward constants, which fails (a) and (T) at a new $\theta$. Proposition 0.1 fixes $J$. It does not fix $(a,b)$. For every $C>0$ the pair $\tilde f=2\,\mathrm{arcsinh}(C\sinh(f^{\rm phys}/2))$, $\tilde g=g/C$ gives the same $J$, so it is invisible to deterministic data and to $\sigma_{\rm phys}$. LDB at the level of rates is a modelling axiom. Jump-level data can test it (Theorem 2).

**Counterexamples.** All three use one loop. Nodes 1 and 2 carry $x=c_1$, $y=c_2$, and edges $e_1,e_2$ both run from 1 to 2 with $W_1=w$, $W_2=-w$ and $T=1$, $U=0$. Then $f_1-f_2=2w$, $J_1=k_1(xe^{w/2}-ye^{-w/2})$, $J_2=k_2(xe^{-w/2}-ye^{w/2})$, and $\dot y=J_1+J_2$. Training uses $w\in\Theta_0=[0,w_0]$, and the test uses $w'>w_0$.

(a′) *UDE closure,* $J_i=N(c,\xi_i,\theta)$. For any continuous $z(c,\theta)$ the closure $(J_1^{\rm true}-z,\ J_2^{\rm true}+z)$ gives the same $\dot c$ at every condition, and $\sigma_{\rm phys}=\sigma^{\rm true}-2wz$. Take $z=0$ on $\Theta_0$ and $z>\sigma^{\rm true}/(2w')$ at $w'$. The closure fits the training data exactly, and $\sigma_{\rm phys}<0$ at $w'$. The violation is a circulating current in $\ker S$, so it stays invisible in $\dot c$ even at $w'$. A universal approximator matches this closure on a compact set to any tolerance, and the strict inequality survives.

(b′) *Condition as a feature of $k$.* Keep $f$ fixed by physics and let $k_1=k_1^{\rm true}+\eta(\theta)$ with $\eta=0$ on $\Theta_0$ and $\eta(w')=M>-k_1^{\rm true}$. The model fits the training data and stays in each fibre $\mathfrak G_\theta(S)$, so (a) holds. But $k$ now moves with $\theta$, which breaks the product structure of $\mathfrak G(S)$. The relative error of $J_1$ at $w'$ is $M/k_1^{\rm true}$, which is arbitrary. If $\theta$ is a feature of a learned force, $f=f^{\rm phys}+\eta(\theta)$, the model leaves $\mathfrak G(S)$ and (a) fails as well.

(c′) *Learned force coefficient.* Fix $k_1=1+\delta$, $k_2=1-\delta$ and learn $u$ in $A_i=uW_i$, a learned field strength or harmonic drive coefficient. Then $\dot y(u)=2(x-y)\cosh(uw/2)+2\delta(x+y)\sinh(uw/2)$. Let the data come from $u^*\neq0$ and let $L_\delta(u)=\sum_i(\dot y^i(u)-\dot y^i(u^*))^2$. At $\delta=0$, $L_0$ is even, vanishes at $\pm u^*$ and is positive at $0$ if some $x_i\neq y_i$, so it is not convex. Also $L_0''(\pm u^*)=2w^2\sinh^2(u^*w/2)\sum_i(x_i-y_i)^2>0$. By the implicit function theorem, for small $\delta$ there is a strict local minimizer near $-u^*$. If the vectors $(x_i-y_i)_i$ and $(x_i+y_i)_i$ are not parallel, all residuals vanish only at $u=u^*$, so this minimizer has $L_\delta>0$ and is spurious. In the cut, $u=1$ is fixed and $(k_1,k_2)$ enter linearly. They are identified when $w\neq0$, because the Möbius determinant of $\varphi_2/\varphi_1$ is $-2\sinh w$.

**Status.** The constructions are elementary, and (b′) restates Proposition 4(b). New here: Proposition 0.1 and the reading of "only".

### (v) Boundaries

*State-dependent $k$* (crowding, saturation, exclusion). $\mathfrak G(S)$ becomes infinite-dimensional, with $k_h$ a positive function of the local state, and the learned map becomes $(\xi_h,s_h(c))\mapsto k_w$. Here $s_h$ is a state feature of hyperedge $h$ that is invariant under reversing $h$, such as the edge-mean density. The state $c$ is not a condition, so the premise of (iii) and of Proposition 0.1 that the learned part does not see $\theta$ still holds. Proposition 0.1 applies pointwise in $c$. The same $k_h$ multiplies $a_h$ and $b_h$, so $\log(a_h/b_h)=f_h^{\rm phys}$ is untouched. Property (a) is unchanged, since it uses only $g\ge0$ at the current state. The detailed-balanced states of (a)(1) still do not depend on $k$. Property (c) holds for $k=\sum_j\omega_j\psi_j(\xi,c)$ with fixed features. In (b) the identity $J-J^{\rm true}=(k-k^{\rm true})\varphi$ still holds pointwise. At a new condition the solution may visit states outside the training range. The error of $k$ there is extrapolation in $s$, which the theorem does not cover. A truth whose mobility depends on the state, such as $k^{\rm true}=k(\xi)/(1+\bar\rho/\rho_s)$, therefore lies in $\mathfrak G(S)$. A chart that reads $\xi$ alone misses it, and a chart that also reads $s_h$ can represent it.

*Temperature.* A known law such as $k=mT$ is built into the map. Eyring's $k=mTe^{-\Delta G^\ddagger/T}$ adds one scalar per class. NNLS at each temperature, followed by linear regression of $\log(k/T)$ on $1/T$, keeps both steps convex.

*Learned $\delta\mu$.* The force is computed from the learned potential, so the model lies in the generator set of the learned free energy. Then $\sum_hJ_hf_h^{\rm model}\ge0$ for every learned $\delta\mu$, and (a)(2) holds for the learned $F$. $\sigma_{\rm phys}$ is controlled only as far as $\delta\mu$ is right. Property (b) needs $\delta\mu$ independent of $\theta$ (P1) and correct at the concentrations reached at $\theta'$. A learned activity correction is constrained only on the concentrations seen in training. Property (c) is convex in $k$ for fixed $\delta\mu$. It is not jointly convex in $(k,\delta\mu)$. For each $\delta\mu$ the optimal $k$ solves the NNLS problem of (c), so $k$ can be eliminated and the outer problem is over $\delta\mu$ alone (variable projection). Learned harmonic drive coefficients are part of $f$, so (c′) applies to them.

### Remark: the other results as structure of $\mathfrak G(S)$

The blind spot (Theorem 1) says that density data are constant along cycle directions of the set of rate laws with free $[A]$, and Proposition 1′ exhibits one such fibre. Within $\mathfrak G(S)$, where $[A]$ is fixed by physics, the fibre reduces to $\ker X$ of (iii)(d). Closed-form coarse graining (Theorem 3) maps $\mathfrak G(S)$ at $A=0$ into $\mathfrak G(\bar S)$, with the explicit $\bar k$ and the log-sum-exp $\bar\mu$, and the stationary statistics are exact. For $A\neq0$ the coarse generator again has the coordinates of (i). Its cycle affinities are fixed, and its potential part is a gauge choice. In minimum-dissipation control the cost is $\int\sum_hJ_h^2/\Lambda_h\,dt$, the metric that $k$ fixes on $\mathfrak G(S)$. The Benamou–Brenier value of a convex program bounds this cost from below. We claim no uncertainty relation here.

**Relation to known theory.** The following structures are known: the exponential family of rates under LDB (Seifert 2012, Rao and Esposito 2016), cycle affinities (Schnakenberg 1976), the Hessian geometry of reaction networks (Kobayashi et al. 2022), and the gradient structure of detailed-balanced dynamics (Maas 2011, Mielke 2011). Near equilibrium $\Lambda\to g$, so $g$ is the edge Onsager coefficient and reciprocity holds by construction. At $[A]=0$, $\dot c=-S\Lambda S^\top\mu$ is the gradient flow of $F$ in the metric of $\Lambda$. The cut thus fixes the energy by physics and learns the dissipation. New here: the learning formulation as a map from material descriptors into $\mathfrak G(S)$, the combination of (a) to (d) that it yields at once, Proposition 0.1, and the counterexamples. Each step is elementary.

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

**Model-level realization (Proposition 1′).** Part (a) does not use the rate law. A parametric rate law can resolve part of $\ker S$. For a linear Markov chain on a simple graph with rich initial data, $\dot\rho=\rho Q$ identifies $Q$, and hence $[A]$. In that case the information comes from the kinetic assumption. The densities alone do not carry it. The blind spot is realized in the rate model of §0 when a cycle consists of kinetically indistinguishable channels.

*Example (the driven phosphorylation cycle).* After the chemostatted species are removed, R1 is $X\to X_p$ and R2 is $X_p\to X$. Then $\dot c_{X_p}=\alpha e^{\mu_X}-\beta e^{\mu_{X_p}}$ with $\alpha=k_1e^{A_1/2}+k_2e^{-A_2/2}$ and $\beta=k_1e^{-A_1/2}+k_2e^{A_2/2}$. Fix $\alpha,\beta$ and any cycle affinity $C=A_1+A_2>0$. Solving for $(k_1,k_2)$ gives positive values whenever $\log(\alpha/\beta)<A_1<\log(\alpha/\beta)+C$, and this interval is never empty. So every $C$ fits the same noiseless trajectories, and $\sigma$ is unbounded above on the fitted set. $\square$

**Status.** Known: (a), (b) and (e), from Schnakenberg (1976), Jiang, Qian and Qian (2004), and Rao and Esposito (2016) for $\sigma=-\dot F+\langle J,A\rangle$. Part (d) is the $\ell$-weighted orthogonal decomposition of Kolchinsky, Dechant, Yoshimura and Ito (arXiv:2206.14599, 2022) and Yoshimura et al. (PRR 2023). Its Hessian-geometric form for reaction networks is due to Kobayashi, Loutchko, Kamimura and Sughiyama (PRR 4, 033208, 2022). New here: the observable count in (c), used as an experimental-design rule, and Proposition 1′. Both are elementary.

## Theorem 2. Cramér–Rao bound for the affinity of a near-irreversible hyperedge

**Statement.** Take one hyperedge whose forward and backward events are counting processes with intensities $a(t)$ and $b(t)$ on $[0,T]$, where $\log a=u+x(t)+A/2$, $\log b=u+y(t)-A/2$, $u=\log k$, and $x=\langle\nu^+,\mu\rangle$, $y=\langle\nu^-,\mu\rangle$ are known from the observed concentrations. Let $\bar N_+=\mathbb{E}\int_0^Ta\,dt$ and $\bar N_-=\mathbb{E}\int_0^Tb\,dt$.

(i) If $k$ is known, $I(A)=(\bar N_++\bar N_-)/4$.

(ii) If $k$ is a nuisance, the efficient information is

$$I_{\rm eff}(A)=\frac{\bar N_+\bar N_-}{\bar N_++\bar N_-}\le\min(\bar N_+,\bar N_-),\qquad \operatorname{Var}(\hat A)\ge\frac1{\bar N_+}+\frac1{\bar N_-}.$$

For constant rates and $b\ll a$ this is $\operatorname{Var}(\hat A)\gtrsim 1/(bT)$, with constant 1. The factor 1/4 belongs to case (i), where it multiplies $a+b$ and the bound is mild.

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

**Remarks.** In (ii) the information rate is $ab/(a+b)$, half the harmonic mean. It never exceeds $\ell=\Lambda(a,b)$, which is itself the Poisson log-rate Fisher information averaged along the e-geodesic $a^{1-s}b^s$. An error $\delta A$ shifts the edge contribution $\sigma_h=J_hf_h$ by $J_h\delta A$. So the near-irreversible edge limits the housekeeping readout of Theorem 1(d) to precision $J_h/\sqrt{\bar N_-}$. This links to hidden entropy production in two ways. Esposito (2012) shows that unresolved processes hide entropy production structurally under coarse graining. Here the backward channel is present in the model, but it carries too few events, so the loss is statistical. In the decomposition of Kolchinsky et al. (2022), the housekeeping part is the projection onto nonconservative forces. For a near-irreversible cycle it is dominated by that cycle's affinity, so it inherits the bound. The limit $b\to0$ is the absolutely irreversible case of Murashita, Funo and Ueda (2014).

**Status.** Known: the Fisher matrix of counting processes. New here: (ii)–(v) in the rate parametrization of §0, in particular the role of $k$ as a nuisance and the Gaussian form (v).

## Theorem 3. Closed-form coarse graining

**Setting.** Here $Q$ is an irreducible generator with $Q_{ij}>0\Leftrightarrow Q_{ji}>0$. Distributions are row vectors with $\dot p=pQ$, and $\pi Q=0$. For a partition into blocks $a$, let $P_{ia}=\mathbf 1[i\in a]$, $\bar\pi=\pi P$, and $\Pi_{ai}=\pi_{i|a}=\pi_i/\bar\pi_a$ for $i\in a$, so $\Pi P=I$. Define $\bar Q=\Pi QP$, that is, $\bar Q_{ab}=\sum_{i\in a}\pi_{i|a}\sum_{j\in b}Q_{ij}$, and $J_{ij}=\pi_iQ_{ij}-\pi_jQ_{ji}$. In the form of §0 for a linear chain, $Q_{ij}=k_{ij}e^{(\mu_i-\mu_j)/2+A_{ij}/2}$ with $k$ symmetric and $A$ antisymmetric. If $A=0$, then $\pi_i\propto e^{-\mu_i}$.

**Statement.**

(i) $\bar\pi$ is stationary for $\bar Q$.

(ii) $\bar J_{ab}:=\bar\pi_a\bar Q_{ab}-\bar\pi_b\bar Q_{ba}=\sum_{i\in a,j\in b}J_{ij}$, so $\bar J=PJ$. Away from stationarity the same identity holds for $p$ exactly when $p=\bar p\,\Pi$, that is, under local equilibrium inside blocks.

(iii) $\bar\sigma\le\sigma-\sigma_{\rm intra}\le\sigma$ at stationarity, where $\sigma_{\rm intra}\ge0$ is the entropy production of transitions inside blocks.

(iv) (a) For any $\bar\mu$, $\bar Q_{ab}=\bar k_{ab}e^{(\bar\mu_a-\bar\mu_b)/2+\bar A_{ab}/2}$ with $\bar k_{ab}=\sqrt{\bar Q_{ab}\bar Q_{ba}}$ and $\bar A_{ab}=\log(\bar Q_{ab}/\bar Q_{ba})-(\bar\mu_a-\bar\mu_b)$. The coarse cycle affinities $\langle\bar z,\bar A\rangle$ do not depend on $\bar\mu$. (b) If $A=0$, then with $\bar\mu_a=-\log\sum_{i\in a}e^{-\mu_i}$ we get $\bar A=0$ and

$$\bar k_{ab}=\sum_{i\in a,\,j\in b}k_{ij}\sqrt{\pi_{i|a}\,\pi_{j|b}} .$$

(c) If $A\neq0$, the log-sum-exp $\bar\mu$ is only a gauge choice. A cycle-space representative of $\bar A$ requires the projection of Theorem 1(d) on the coarse graph.

(v) The lag propagator, stated below.

(vi) Coarse graining along nested partitions composes, $\mathrm{RG}_{\mathcal B}\circ\mathrm{RG}_{\mathcal A}=\mathrm{RG}_{\mathcal B\circ\mathcal A}$.

**Proof.** (i) $\bar\pi\Pi=\pi$, so $\bar\pi\bar Q=\pi QP=0$. (ii) $\bar\pi_a\bar Q_{ab}=\sum_{i\in a,j\in b}\pi_iQ_{ij}$. The transient statement follows by replacing $\pi_{i|a}$ with $p_i/\bar p_a$. (iii) Write $\sigma=\sum_{i\neq j}x_{ij}\log(x_{ij}/x_{ji})$ with $x_{ij}=\pi_iQ_{ij}$. For each ordered block pair $a\neq b$, the log-sum inequality gives $\sum_{i\in a,j\in b}x_{ij}\log\frac{x_{ij}}{x_{ji}}\ge X_{ab}\log\frac{X_{ab}}{X_{ba}}$ with $X_{ab}=\bar\pi_a\bar Q_{ab}$. The pairs inside blocks contribute $\tfrac12\sum(x_{ij}-x_{ji})\log(x_{ij}/x_{ji})\ge0$. (iv) Part (a) is the split of $\log\bar Q$ into symmetric and antisymmetric parts. Around a coarse cycle the $\bar\mu$ terms cancel. For (b), $A=0$ gives $\pi_iQ_{ij}=\pi_jQ_{ji}$, and summing over $i\in a,\,j\in b$ shows that $\bar Q$ is reversible with respect to $\bar\pi$. Then $\log(\bar Q_{ab}/\bar Q_{ba})=\log(\bar\pi_b/\bar\pi_a)=\bar\mu_a-\bar\mu_b$, because $\bar\pi_a\propto\sum_{i\in a}e^{-\mu_i}=e^{-\bar\mu_a}$. Next, $\pi_iQ_{ij}=Z^{-1}k_{ij}e^{-(\mu_i+\mu_j)/2}$ and $\sqrt{\bar\pi_a\bar\pi_b}=Z^{-1}e^{-(\bar\mu_a+\bar\mu_b)/2}$, and dividing gives $\bar k$. (vi) This follows from $\bar\pi\Pi=\pi$ and the nesting of the conditional weights. $\square$

**Lag propagator (v).** Let $\bar T(\tau)=\Pi e^{\tau Q}P$ (in the column convention, $Pe^{\tau L}\Pi$). Then $\bar\pi\bar T(\tau)=\pi e^{\tau Q}P=\bar\pi$, and $\bar\pi_a\bar T_{ab}(\tau)=\Pr_\pi(X_0\in a,X_\tau\in b)$. So the stationary two-time block statistics and the lagged net fluxes are exact, and $\bar T(\tau)=I+\tau\bar Q+O(\tau^2)$. But $\bar T$ need not satisfy Chapman–Kolmogorov, and $\tau^{-1}\log\bar T(\tau)$ need not be a generator.

*Counterexample.* Take a ring $1\to2\to3\to1$ with forward rate $\gamma$ and backward rate $\epsilon$, so $\pi$ is uniform. Use the blocks $\{1\}$ and $\{2,3\}$. At $\epsilon=0$ the displacement is $N\bmod3$ with $N\sim\mathrm{Poisson}(\lambda)$ and $\lambda=\gamma\tau$, and $p_0=\Pr(N\equiv0)=\tfrac13[1+2e^{-3\lambda/2}\cos(\sqrt3\lambda/2)]$. A direct count gives $\operatorname{tr}\bar T=(1+3p_0)/2$. At $\lambda=2\pi/\sqrt3$, $p_0=\tfrac13(1-2e^{-\sqrt3\pi})\approx0.3304$, so $\det\bar T=\operatorname{tr}\bar T-1\approx-0.0043<0$. A real $2\times2$ matrix with one negative eigenvalue has no real logarithm (Culver 1966), so no generator gives $\bar T(\tau)$. The strict inequality persists for small $\epsilon>0$. The hidden circulation inside $\{2,3\}$ is the memory that a Markov generator cannot hold. $\square$

**Where Chentsov's theorem enters.** Parts (i), (ii), (iv), (v) and (vi) are algebra. Part (iii) is a data-processing inequality for the KL divergence on transition pairs, the monotonicity of Csiszár. It does not use Chentsov's theorem. Chentsov's theorem (Čencov 1982, Campbell 1986) is a uniqueness statement. Up to scale, the Fisher–Rao metric is the only Riemannian metric on simplices that is invariant under all congruent Markov embeddings, and the $\alpha$-connections are the only invariant affine connections (Amari and Nagaoka 2000). Our $\Pi$ is such an embedding. Wolfer and Watanabe (arXiv:2203.04575) build the kernel version, in which congruent embeddings are linear right inverses of lumping. The local-equilibrium set $\{\bar p\,\Pi\}$ is both e-flat and m-flat. The map $p\mapsto(pP)\Pi$ is the m-projection onto it, and the Pythagorean relation holds. So the claim "the coarse model carries the same geometry as the fine one, and no other geometry does this for every partition" depends on Chentsov's theorem. The formulas for $\bar Q$, $\bar\mu$ and $\bar k$ do not. The edge metric $\ell$ and the Maas metric are not covered by Chentsov's theorem, and we make no uniqueness claim for them. Closure of the exponential family is trivial for the full family on a finite set. The content is the explicit map of natural parameters, namely the block log-partition function $\bar\mu_a$, with the block label as the sufficient statistic of the local-equilibrium family. Subfamilies are not closed. For example, $\mu=U/T$ maps to $\bar\mu_a=-\log\sum_{i\in a}e^{-U_i/T}$, which is not affine in $1/T$. When $A=0$, $\bar Q$ is reversible, so the coarse model is again a gradient flow of $\mathrm{KL}(\cdot\|\bar\pi)$ in the Maas metric of the coarse graph (Maas 2011). That coarse metric is not the pushforward of the fine one.

**Status.** Known: (i)–(iii), from Esposito (2012), local-equilibrium projection in Markov state models (Hummer and Szabo 2015), and the log-sum inequality. New here: the closed form (iv)(b) in the parameters of §0, the gauge statement (iv)(c), the explicit counterexample in (v), and the account of what does and does not depend on Chentsov's theorem.

## Proposition 4. Extrapolation in the conditions

**Statement.** Assume the following.

- (P1) $\mu=\log c+U/T+\delta\mu(c)$, where $\delta\mu$ does not depend on the conditions.
- (P2) $A_h=W_h(F)/T+A_h^{\rm chem}$, where $W_h$ is the line integral of the force $F$ along $h$ and the chemostat part is $-\sum_X S_{Xh}(\log c_X+U_X/T)$.
- (P3) $k_h=T\,m_h$ with $m_h$ independent of the conditions (the Einstein relation $D=mT$).

Then $\log a_h$ and $\log b_h$ are affine in the natural parameters $(1/T,\,F/T,\,\log c_{\rm chem})$. A change of the conditions, monotone or not, therefore maps the model by substitution, with $m$ and $\delta\mu$ held fixed. For $T\to T'=rT$,

$$k\to r\,k,\qquad U/T\to (U/T)/r,\qquad W/T\to (W/T)/r,\qquad \log c,\ \log c_X\ \text{unchanged}.$$

So the rate prefactor scales by $r$, and every non-entropic force, including $A$, scales by $1/r$. The quantity that scales is the prefactor $k\propto D=mT$. The Einstein mobility $m$ stays fixed. A drive edit $F\to sF$ gives $A\to sA$ on the force part.

(a) If the true system satisfies (P1)–(P3), the log-rate error of a fitted model is the same at every condition, because the condition terms are exact.

(b) If a condition enters as a normalized feature of a universal approximator, data on a training set $K$ of condition values do not constrain predictions at $\kappa'\notin\bar K$. For any continuous target there is a network in the class that fits the training data to any tolerance and matches the target at $\kappa'$. ReLU networks also extrapolate linearly in the input (Xu et al. 2021). That extrapolation is linear in the rate. The physics is linear in the log-rate.

**Proof.** (P1)–(P3) give $\log a_h=\log m_h+\log T+\langle\nu_h^+,\log c+\delta\mu\rangle+T^{-1}\big[\langle\nu_h^+,U\rangle+W_h/2\big]+\tfrac12A_h^{\rm chem}$, and similarly for $b_h$. Every condition appears explicitly. For transport, the drift $k(W-\Delta U)/T=m(W-\Delta U)$ does not depend on $T$, and the diffusion $k=mT$ does. This is the SQRA discretization of $\partial_t\rho=\nabla\cdot(mT\nabla\rho+m\rho(\nabla U-F))$. (b) follows from the Tietze extension theorem and universal approximation on a compact set containing $K\cup\{\kappa'\}$. $\square$

**Caveats.** For reactions with a barrier, Eyring's law $k_h\propto T e^{-\Delta G_h^\ddagger/T}$ needs one more scalar per hyperedge class, which data at two temperatures can identify. Debye–Hückel corrections depend on $T$, so (P1) must then be written with the explicit $T$ dependence. After coarse graining, temperature enters through $\bar\mu_a=-\log\sum_{i\in a}e^{-U_i/T}$, which is not affine in $1/T$. The map stays explicit, but it is no longer a rescaling.

**Status.** Known: local detailed balance with the conditions entering as energies divided by $T$, and the Einstein relation (Seifert 2012, Rao and Esposito 2016). New here: the exact edit map for the model class with forces fixed by physics, and the non-identifiability contrast (b), which is elementary.

## Information geometry of the chart

**Definition (Neural Chart).** Let $\Theta$ be the space of conditions and $\Xi$ the space of local material descriptors. A Neural Chart is the map

$$\Phi_w:\Theta\times\Xi^m\to\mathfrak G(S),\qquad(\theta,\xi)\mapsto\big(f^{\rm phys}(\cdot;\theta),\,k_w(\xi)\big).$$

The physical coordinate $f^{\rm phys}$ is fixed by the axioms of §0 and Theorem 0(ii). The material coordinate $k_w(\xi)$ is the output of a network. For state-dependent mobility, $k_w$ also reads the state feature $s_h(c)$ of Theorem 0(v). Strictly, $\Phi_w$ is a parametrization of $\mathfrak G(S)$, the inverse of a coordinate chart. The axioms supply some coordinates of the generator and a network supplies the rest. A Neural Chart is any model built this way.

The results below restate Theorems 0 to 3 and Proposition 4 in the dually flat geometry of the local jump law and prove them there. Throughout, $L(a,b)$ is the local law of the event counts at a frozen state. On a window of length $\tau$ the forward and backward counts of hyperedge $h$ are independent Poisson variables with means $\tau a_h$ and $\tau b_h$. We quote divergences per unit time. For intensity vectors, $D(a\|a')=\sum_h(a_h\log(a_h/a'_h)-a_h+a'_h)$.

### IG1. Entropy production is a path divergence

**Statement.** (a) Let $P_{[0,T]}$ be the path measure of a Markov jump process with rates $w_{xy}$ started at $p_0$. Let $\tilde P_{[0,T]}$ be the law of the reversed path $\omega(T-\cdot)$ when $\omega$ follows the same generator started at $p_T$. Then

$$D(P_{[0,T]}\|\tilde P_{[0,T]})=\int_0^T\sigma_{\rm tot}(t)\,dt,\qquad \sigma_{\rm tot}=\sum_{x<y}(p_xw_{xy}-p_yw_{yx})\log\frac{p_xw_{xy}}{p_yw_{yx}} .$$

At stationarity $\sigma=T^{-1}D(P_{[0,T]}\|\tilde P_{[0,T]})$ for every $T$. Otherwise $\sigma=\lim_{T\to\infty}T^{-1}D$, since the boundary term is bounded.

(b) At a frozen state,

$$D\big(L(a,b)\,\|\,L(b,a)\big)=\sum_h(a_h-b_h)\log\frac{a_h}{b_h}=\sum_h\big[D(a_h\|b_h)+D(b_h\|a_h)\big].$$

Each term is the Jeffreys divergence between the Poisson laws of intensities $a_h$ and $b_h$. For a linear network, with $a_h=p_iQ_{ij}$, this is $\sigma_{\rm tot}$ of (a). For mass-action networks with extensive propensities, $\sigma_{\rm tot}/\Omega$ of the chemical master equation converges to the §0 expression along the mean-field solution.

**Proof.** (a) The log-likelihood ratio of two jump processes is the sum of log rate ratios over jumps minus the integrated difference of escape rates (Andersen et al. 1993). The escape terms cancel because the reversed path visits the same states for the same times. What remains is $\log p_0(\omega_0)-\log p_T(\omega_T)+\sum_{\rm jumps}\log(w_{xy}/w_{yx})$. Its expectation is the change of Shannon entropy plus the medium term, together $\int\sigma_{\rm tot}\,dt$. (b) The Poisson divergence is $a\log(a/b)-a+b$, and reversal swaps the two intensities. $\square$

**Status.** Known: Lebowitz and Spohn (1999), Maes (1999), Maes and Netočný (2003), Gaspard (2004), Seifert (2012). The macroscopic limit is due to Ge and Qian (2016) and Rao and Esposito (2016). It follows that the readout $\sigma$ is a property of the learned path measure.

### IG2. Dual flatness, and the conditions as natural parameters

**Statement.** (a) $\{L(a,b)\}$ is an exponential family with natural coordinates $(\log a,\log b)$, expectation coordinates $(a,b)$ and potential $\psi=\sum_h(a_h+b_h)$. The linear change $f_h=\log a_h-\log b_h$, $\lambda_h=\tfrac12(\log a_h+\log b_h)=\log g_h$ gives e-coordinates $(f,\lambda)$, in which $\psi=\sum_h2e^{\lambda_h}\cosh(f_h/2)$. The dual m-coordinates are

$$\partial_{f_h}\psi=J_h/2,\qquad\partial_{\lambda_h}\psi=a_h+b_h=:\tau_h ,$$

the current and the traffic. The Fisher metric of hyperedge $h$ in $(f_h,\lambda_h)$ is

$$G_h=\begin{pmatrix}\tau_h/4&J_h/2\\ J_h/2&\tau_h\end{pmatrix},$$

so $f$ and $\lambda$ are Fisher-orthogonal exactly when $J_h=0$.

(b) In the chart, $f=f^{\rm phys}(c;\theta)$ and $\lambda=\log k+\log\varepsilon(c;\theta)$. By Proposition 4 both are affine in $\eta(\theta)=(1/T,F/T,\mu_Y)$ at fixed $c$, up to the known shift $\log T$ when $k=mT$. So at fixed $c$ and $k$ the conditions are natural parameters,

$$e(\theta)=e_0(c)+B(c)\,\eta(\theta)+(0,\log k),$$

and the model family over $\Theta$ is e-flat. A change of condition is the e-translation by the known vector $B(c)\Delta\eta$. Model and truth differ by the e-vector $(0,\log(k/k^{\rm true}))$ at every condition, which is Theorem 0(iii)(b) in e-coordinates.

(c) *Extrapolation error in the Fisher and Onsager geometry.* Let $\delta_h=k_h/k_h^{\rm true}-1$, $\delta_k=\max_h|\delta_h|$, and evaluate $\tau,\Lambda,\sigma$ at the truth at any $(c,\theta')$. Then the following hold.

(c.1) $D(L_{\rm model}\|L_{\rm true})=\sum_h\tau_h^{\rm true}\,\phi(\delta_h)\le\bar\phi(\delta_k)\,\tau^{\rm true}$, where $\phi(\delta)=(1+\delta)\log(1+\delta)-\delta=\delta^2/2+O(\delta^3)$ and $\bar\phi(\delta)=\max\{\phi(\delta),\phi(-\delta)\}$.

(c.2) $\|J-J^{\rm true}\|^2_{\Lambda^{-1}}=\sum_h\delta_h^2\,\sigma_h^{\rm true}$.

(c.3) With the Onsager dual norm $\|v\|_*^2=v^\top(S\Lambda S^\top)^+v$ on $\operatorname{im}S$, for which $\|\dot c\|_*^2=\sigma_{\rm ex}$ by Theorem 1(d),

$$\|\dot c-\dot c^{\rm true}\|_*\ \le\ \|J-J^{\rm true}\|_{\Lambda^{-1}}\ \le\ \delta_k\sqrt{\sigma^{\rm true}(c;\theta')} .$$

If $\delta_h\equiv\delta$, then $\|\dot c-\dot c^{\rm true}\|_*=|\delta|\sqrt{\sigma^{\rm true}_{\rm ex}}$ exactly.

Since $\tau_h=2k_h\varepsilon_h\cosh(f_h/2)$ and $\sigma_h=2k_h\varepsilon_hf_h\sinh(f_h/2)$, the Euclidean error of Theorem 0(iii)(b) and the divergence grow like $e^{|f_h|/2}$. The Onsager error grows only like $\sqrt{f_h\sinh(f_h/2)}\sim e^{|f_h|/4}$. The normalized errors $D/\tau^{\rm true}$ and $\|\Delta\dot c\|_*/\sqrt{\sigma^{\rm true}}$ stay below $\bar\phi(\delta_k)$ and $\delta_k$ at every condition.

**Proof.** (a) The log-likelihood is $\sum_h(N_h^+\log a_h+N_h^-\log b_h)-\tau\psi$ plus a term free of parameters. Differentiating $\psi$ once and twice in $(f,\lambda)$ gives the m-coordinates and $G_h$. (b) follows from Proposition 4 and Theorem 0(ii). (c.1) Both intensities scale by $1+\delta_h$, and $\phi$ is convex with minimum $0$ at $0$. (c.2) $J_h-J_h^{\rm true}=\delta_hJ_h^{\rm true}$ by Theorem 0(iii)(b), and $(J_h^{\rm true})^2/\Lambda_h=J_hf_h=\sigma_h$. (c.3) $\Delta\dot c=S\Delta J$, and $\|SJ'\|_*\le\|J'\|_{\Lambda^{-1}}$ by the variational formula of Theorem 1(d). For uniform $\delta$, $\Delta\dot c=\delta\dot c^{\rm true}$. $\square$

**Status.** Known: (a). Up to a constant, $\psi$ is the dual dissipation potential of the cosh gradient structure (Mielke, Peletier and Renger 2014). See also Maes and Netočný (2008) and Kaiser, Jack and Zimmer (2018). New here: (b) and (c), both elementary.

### IG3. The Onsager metric is an averaged Fisher information

**Statement.** The Fisher information of a Poisson law for its log-intensity is the intensity. Along the e-geodesic $\lambda_s=a^{1-s}b^s$ from $a$ to $b$,

$$\Lambda(a,b)=\int_0^1\lambda_s\,ds,\qquad a-b=\Lambda(a,b)\log\frac ab,\qquad \sigma_h=\Lambda_hf_h^2=D(a_h\|b_h)+D(b_h\|a_h).$$

Moreover $g=\lambda_{1/2}\le\Lambda\le(a+b)/2$, and all three equal the Fisher information $a$ at equilibrium. In the chart $\Lambda_h=k_h\cdot\varepsilon_h\,\frac{\sinh(f_h/2)}{f_h/2}$. The material sets the Fisher information at the e-midpoint. The statement concerns one channel. In a network the channels can be correlated.

**Proof.** $\int_0^1a(b/a)^sds=(b-a)/\log(b/a)$. Along the geodesic $d\lambda_s/ds=-f\lambda_s$, and integration gives $a-b=f\int_0^1\lambda_sds$. This is the mean-value form of the Legendre map of a dually flat family, and the Jeffreys divergence is the pairing of the two differences. The inequalities are the Hermite–Hadamard inequality for the convex map $s\mapsto\lambda_s$. The chart form is Theorem 0(i). $\square$

**Status.** Known: the integral form (Carlson 1972). Its reading follows from the dually flat structure (Amari and Nagaoka 2000). New here: the factorization in the chart.

### IG4. The Pythagorean split as e- and m-projections

**Statement.** (a) *Exact quadratic split.* At every state $\sigma=\|f\|_\Lambda^2$, because $J=\Lambda f$ exactly. The $\Lambda$-orthogonal split $f=f_{\rm ex}+f_{\rm hk}$ of Theorem 1(d), with $f_{\rm ex}\in\operatorname{im}S^\top$ and $\Lambda f_{\rm hk}\in\ker S$, gives $\sigma=\sigma_{\rm ex}+\sigma_{\rm hk}$ exactly. It is the Pythagorean theorem in the tangent space, in the averaged Fisher metric of IG3 frozen at the current state. It is exact and does not rely on linear response. Only $\Lambda$-orthogonality makes the split additive, since any other split leaves the cross term $2\langle f_1,f_2\rangle_\Lambda$. For example, at a steady state $f=\Lambda^{-1}J$ with $J\in\ker S$. This $f$ is Euclidean-orthogonal to $\operatorname{im}S^\top$ only when $\Lambda$ is constant on the support of $J$, so a Euclidean projection books part of $\sigma$ as excess.

(b) *Exact divergence split* (Kolchinsky et al. 2022). Tilt the one-way intensities as $j^+_h(\vartheta)=a_he^{\vartheta^+_h-f_h}$, $j^-_h(\vartheta)=b_he^{\vartheta^-_h+f_h}$. Then $\vartheta=(f,-f)$ gives $P=L(a,b)$ and $\vartheta=0$ gives $\tilde P=L(b,a)$. The conservative parameters $\mathcal C=\{(-S^\top\varphi,S^\top\varphi)\}$ form a linear subspace of natural parameters, an e-flat submanifold that contains $\tilde P$. Let $Q^*$ be the m-projection of $P$ onto $\mathcal C$. Then

$$\sigma=D(P\|\tilde P)=\underbrace{D(P\|Q^*)}_{\sigma^{\rm IG}_{\rm hk}}+\underbrace{D(Q^*\|\tilde P)}_{\sigma^{\rm IG}_{\rm ex}},\qquad \sigma^{\rm IG}_{\rm ex}=\min\{D(j\|\tilde P):S(j^+-j^-)=\dot c\}.$$

The second formula says that $Q^*$ is also the e-projection of $\tilde P$ onto the m-flat set of laws with the observed $\dot c$. Both statements are exact far from equilibrium.

(c) *Relation.* $\sigma^{\rm IG}_{\rm hk}\le\sigma_{\rm hk}$ and $\sigma^{\rm IG}_{\rm ex}\ge\sigma_{\rm ex}$ (Kolchinsky et al. 2022, Eq. S79). Both housekeeping terms vanish iff $f\in\operatorname{im}S^\top$, and both excess terms vanish at a steady state. They agree to second order in small forces. Part (a) is exact in a metric and part (b) is exact in divergences. The two are different decompositions.

(d) *Not a projection onto $A=0$.* The reversible subfamily $\{[A]=0\}$ of the chart is also e-flat, and the m-projection of $P$ onto it keeps $\dot c$ and the traffic $\tau_h$. But $\tilde P$ lies in that family only when $f\in\operatorname{im}S^\top$, so its Pythagorean relation does not split $\sigma$.

**Proof.** (a) Theorem 1(d). (b) The tilted laws form an exponential family in $\vartheta$, and the generalized KL is its Bregman divergence. The Pythagorean theorem for the m-projection onto an e-flat set (Amari and Nagaoka 2000, Csiszár 1975), applied with $R=\tilde P\in\mathcal C$, gives the split. (c) is cited. (d) The stationarity conditions of $\min_RD(P\|R)$ over $\log(a'/b')\in\operatorname{im}S^\top$ with free $\lambda'$ are $a'_h+b'_h=a_h+b_h$ and $S(J'-J)=0$. $\square$

**Status.** Known: (a) is the Onsager-projective decomposition (Dechant, Sasa and Ito 2022, Yoshimura et al. 2023), which Theorem 1(d) states. Parts (b) and (c) are due to Kolchinsky et al. (2022). New here: the reading of (a) through IG3, and (d).

### IG5. The lower bound on entropy production as an information projection

**Statement.** Let the one-way intensities be linear in parameters $r\ge0$, $a_h=\langle\alpha_h,r\rangle$, $b_h=\langle\beta_h,r\rangle$, with known nonnegative exposures $\alpha_h(c,\theta),\beta_h(c,\theta)$. This covers the cut ($r=k$) and rate models in which several one-way rates share a parameter. Let $\mathcal F$ be a convex set of parameters consistent with the data, such as $\{r:\mathcal L(r)\le(1+\epsilon)\mathcal L^*\}$ for a convex loss. Let $\bar\sigma(r)=\sum_iw_i\sum_h(a_h-b_h)\log(a_h/b_h)$ over evaluation states $(c^i,\theta^i)$.

(a) $\bar\sigma$ is convex in $r$, and $\sigma_{\min}(\mathcal F)=\inf_{\mathcal F}\bar\sigma$ is a convex program whose local minimizers are global.

(b) $\bar\sigma(r)=\sum_iw_iD(L^i_r\|\tilde L^i_r)$, so $\sigma_{\min}$ is the smallest divergence between a law and its reverse among the laws consistent with the data. If the truth lies in $\mathcal F$, then $\sigma^{\rm true}\ge\sigma_{\min}$.

(c) If $\mathcal F'\subset\mathcal F$, then $\sigma_{\min}(\mathcal F')\ge\sigma_{\min}(\mathcal F)$. An extra observable shrinks $\mathcal F$, so the bound can only tighten.

(d) If $\mathcal F$ allows $b_h\to0$ with $a_h$ bounded below at a weighted state, then $\sup_{\mathcal F}\bar\sigma=\infty$. With $(a,b)$ free apart from $S(a-b)=\dot c$ the infimum would be $0$, since $J\log(1+J/b)\to0$ as $b\to\infty$. The bound draws its content from the exposures that physics fixes.

**Proof.** $(u,v)\mapsto u\log(u/v)$ is the perspective of $t\log t$, hence jointly convex, and $(u-v)\log(u/v)=u\log(u/v)+v\log(v/u)$. Linear maps keep convexity. (b) is IG1(b), (c) is an infimum over a subset, and in (d) the term of the vanishing $b_h$ diverges. $\square$

**Status.** Known: joint convexity of relative entropy (Csiszár 1967, Cover and Thomas 2006). The projection reading follows Csiszár (1975) in an extended sense. It is not the I-projection of a fixed reference measure, because both arguments move with $r$. In the cut, $\bar\sigma$ is linear in $k$. New here: the data-consistent bound and (d).

### IG6. Coarse graining is data processing

**Statement.** Let $\Phi$ lump states as in Theorem 3, applied to whole paths.

(a) $\Phi$ commutes with time reversal, so at stationarity $D(\Phi_*P_{[0,T]}\|\Phi_*\tilde P_{[0,T]})\le T\sigma$.

(b) The same argument on transition pairs $(X_t,X_{t+dt})$ gives $\bar\sigma\le\sigma$ for the Markov surrogate $\bar Q=\Pi QP$, whose entropy production is the divergence rate of block transition pairs. Theorem 3(iii) sharpens this to $\bar\sigma\le\sigma-\sigma_{\rm intra}$.

(c) $\bar\pi=\pi P$ and $\bar J=PJ$ are algebra.

(d) Chentsov's theorem supports one sentence only. Up to scale the Fisher metric is the unique metric invariant under congruent Markov embeddings such as $\Pi$, so the embedded coarse simplex carries the fine Fisher geometry isometrically and no other metric does this for every partition. Parts (a) to (c) do not use it, and it does not cover $\Lambda$.

**Proof.** (a) The data-processing inequality for a measurable map (Csiszár 1967), with $\Phi\circ\Theta=\Theta\circ\Phi$ for the reversal $\Theta$. (b) The pair law has off-diagonal mass $\pi_iQ_{ij}dt$, and its divergence from the reversed pair law is $\sigma\,dt+O(dt^2)$. Lumping maps it to $X_{ab}dt$ and gives $\bar\sigma\,dt$. (c) and (d) are Theorem 3. $\square$

**Status.** Known: Kawai, Parrondo and Van den Broeck (2007), Gomez-Marin, Parrondo and Van den Broeck (2008), Roldán and Parrondo (2010), Esposito (2012). The account of Chentsov's theorem is that of Theorem 3.

### IG7. Fisher bounds on identifiability, and two sources of blindness

**Statement.** (a) *Information.* With $k$ a nuisance, jump data give $\operatorname{Var}(\hat A)\ge1/\bar N_++1/\bar N_-$ (Theorem 2). The efficient information rate $ab/(a+b)$ is at most $g/2$, half the Fisher information at the e-midpoint of IG3, and it tends to $0$ with $b$.

(b) *Topology.* For density data $\dot c=SJ$ with noise of intensity $\eta^2$, the Fisher information of $k$ in the cut is $X^\top X/\eta^2$. Its null space $\ker X=\{v:\operatorname{diag}(\varphi^i)v\in\ker S\ \forall i\}$ carries zero information at any sample size (Theorem 0(iii)(d)).

(c) So non-identifiability has two sources. The topological one gives exact zero eigenvalues from $\ker S$, such as parallel channels. Only a new observable outside $\operatorname{im}S^\top+\operatorname{span}\{w_r\}$ removes it (Theorem 1(c)). The informational one gives positive eigenvalues that decay with the backward rate. Only more events remove it, about $(1+e^{A})/\varepsilon^2$ forward events for error $\varepsilon$.

**Proof.** (a) Theorem 2(ii) and $a+b\ge2\sqrt{ab}$. (b) is the Fisher information of a linear Gaussian model. (c) combines the two. $\square$

**Status.** Known: (a) is Theorem 2 and (b) is standard. New here: placing both in the spectrum of one Fisher matrix.

### The division of labour between information geometry and Hodge theory

The Fisher geometry supplies the metric, the projections, the lower bounds and the coordinates in which the conditions act. Because the conditions are natural parameters, a new condition is an e-translation, and its error is controlled in the Onsager metric (IG2, IG3). The Hodge structure supplies the coboundary $S^\top$ that turns potentials into forces, the cycle space $\ker S$, the blind spot of densities, and the dimension $\beta_1$ of the non-equilibrium drive, whose classes $[A]$ live in $(\ker S)^*$. The two meet in the $\Lambda$-weighted Hodge decomposition. There the averaged Fisher metric makes $\operatorname{im}S^\top$ and $\Lambda^{-1}\ker S$ orthogonal, and $\sigma$ splits exactly. We derive no thermodynamic uncertainty relation here. Every bound in this section is either a convex lower bound or a Cramér–Rao bound.

## References

Amari and Nagaoka, *Methods of Information Geometry*, AMS (2000). Andersen, Borgan, Gill and Keiding, *Statistical Models Based on Counting Processes*, Springer (1993). Brunton, Proctor and Kutz, PNAS 113, 3932 (2016). Bruggeman, Burchard, Kooi and Sommeijer, Appl. Numer. Math. 57, 36 (2007). Burchard, Deleersnijder and Meister, Appl. Numer. Math. 47, 1 (2003). Campbell, Proc. AMS 98, 135 (1986). Carlson, Amer. Math. Monthly 79, 615 (1972). Čencov, *Statistical Decision Rules and Optimal Inference*, AMS (1982). Cover and Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006). Csiszár, Studia Sci. Math. Hungar. 2, 299 (1967). Csiszár, Ann. Probab. 3, 146 (1975). Culver, Proc. AMS 17, 1146 (1966). Dechant, Sasa and Ito, PRR 4, L012034 (2022). Esposito, PRE 85, 041125 (2012). Frank and Wolfe, Naval Res. Logist. Q. 3, 95 (1956). Gaspard, J. Stat. Phys. 117, 599 (2004). Ge and Qian, PRE 94, 052150 (2016). Gomez-Marin, Parrondo and Van den Broeck, PRE 78, 011107 (2008). Horn and Jackson, Arch. Ration. Mech. Anal. 47, 81 (1972). Hummer and Szabo, J. Phys. Chem. B 119, 9029 (2015). Jiang, Qian and Qian, *Mathematical Theory of Nonequilibrium Steady States*, LNM 1833 (2004). Kaiser, Jack and Zimmer, J. Stat. Phys. 170, 1019 (2018). Kawai, Parrondo and Van den Broeck, PRL 98, 080602 (2007). Kobayashi, Loutchko, Kamimura and Sughiyama, PRR 4, 033208 (2022). Kolchinsky, Dechant, Yoshimura and Ito, arXiv:2206.14599 (2022). Lawson and Hanson, *Solving Least Squares Problems*, Prentice-Hall (1974). Lebowitz and Spohn, J. Stat. Phys. 95, 333 (1999). Maas, J. Funct. Anal. 261, 2250 (2011). Maes, J. Stat. Phys. 95, 367 (1999). Maes, Phys. Rep. 850, 1 (2020). Maes and Netočný, J. Stat. Phys. 110, 269 (2003). Maes and Netočný, EPL 82, 30003 (2008). Mielke, Nonlinearity 24, 1329 (2011). Mielke, Peletier and Renger, Potential Anal. 41, 1293 (2014), arXiv:1312.7591. Murashita, Funo and Ueda, PRE 90, 042110 (2014). Peletier and Schlichting, arXiv:2203.05435 (2022). Rao and Esposito, PRX 6, 041064 (2016). Roldán and Parrondo, PRL 105, 150607 (2010). Schnakenberg, Rev. Mod. Phys. 48, 571 (1976). Seifert, Rep. Prog. Phys. 75, 126001 (2012). Wolfer and Watanabe, arXiv:2106.05669 (2021), and arXiv:2203.04575. Xu et al., ICLR (2021). Yoshimura, Kolchinsky, Dechant and Ito, PRR 5, 013017 (2023).
