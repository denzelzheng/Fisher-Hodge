# E4: closed-form coarse-graining (RG) of the learned TC-ME generator vs coarse-grid training (S2, 2 seeds)

RG rule: Qbar_ab = sum_{i in a} (pi_i/pi_a) sum_{j in b} Q_ij (projected generator, Esposito 2012) with pi = stationary state of the trained 32x32 TC-ME for the given U, dpsi; rho-dependence of g via the local-equilibrium closure rho_i = 4 rhobar_a pi_i/pi_a. Parametric form: phibar = -log sum exp(-phi) (block), kbar = sqrt(Qbar_ab Qbar_ba), Abar = log(Qbar_ab/Qbar_ba) - (phibar_a - phibar_b). No coarse data. Scratch models: same protocol as E1 (2000 iters) on the 2x2-coarsened (or 4x4) training trajectories, lumped U/psin/dpsi as in coarse_test.py. Coarse reference = 2x2 block mean of the 32x32 data (itself a 64x64 FV solve, block-averaged). All numbers: median over the 16 test trajectories, mean±std over 2 seeds.

## Rollout accuracy on the 16x16 coarse test set (relative L2 vs block-averaged reference)

| model | coarse training data | T ID | 2T ID | 10T ID | 10T OOD-pot | T OOD-pot | max mass err | min rho | traj with rho<0 |
|---|---|---|---|---|---|---|---|---|---|
| (a) RG-TC-ME (closure; the rule) | none | 0.305±7.5e-05 | 0.164±2.91e-05 | 0.00298±8.83e-05 | 0.0154±0.000381 | 0.269±0.000343 | 3.5e-06 | 5.3e-05 | 0/64 |
| (a) RG-TC-ME (generator frozen at Q(pi)) | none | 0.304±8.41e-05 | 0.164±2.79e-05 | 0.00298±8.83e-05 | 0.0154±0.000381 | 0.267±0.000332 | 3.5e-06 | 5.3e-05 | 0/64 |
| (a) diagnostic: closure with kbar/2 (NOT the rule, breaks flux exactness) | none | 0.0851±4.85e-05 | 0.0626±1.22e-05 | 0.00382±0.000189 | 0.0173±0.000174 | 0.167±0.000102 | 1.1e-06 | 5.0e-05 | 0/64 |
| (b) fine TC-ME 32x32 rollout, then lump (E2) | (fine data) | 0.00329±7.05e-05 | 0.00291±1.21e-05 | 0.00309±0.000124 | 0.0154±6.48e-05 | 0.0128±0.000179 | 6.0e-07 | 1.5e-05 | 0/64 |
| (b) fine Fisher-Hodge 32x32 rollout, then lump (E2) | (fine data) | 0.00356±7.84e-05 | 0.00332±1.17e-06 | 0.00304±0.000177 | 0.0134±0.000482 | 0.0137±0.00091 | 3.2e-07 | 2.0e-05 | 0/64 |
| (b) fine Flux-GNN 32x32 rollout, then lump (E2) | (fine data) | 0.00224±3.91e-05 | 0.00231±0.000173 | 0.00239±0.000263 | 0.0167±0.0017 | 0.0167±0.000277 | 4.2e-07 | 1.4e-05 | 0/64 |
| (c) TC-ME trained from scratch on 16x16 | 64 coarse traj | 0.0139±4.65e-05 | 0.0121±1.71e-05 | 0.0127±6.45e-06 | 0.0634±0.000155 | 0.0597±4.11e-05 | 1.3e-06 | 2.3e-05 | 0/64 |
| (c) Fisher-Hodge trained from scratch on 16x16 | 64 coarse traj | 0.0152±5.48e-05 | 0.0137±0.000162 | 0.0132±0.000148 | 0.0517±0.000825 | 0.0602±0.000484 | 1.1e-06 | 3.5e-05 | 0/64 |
| (c) Flux-GNN trained from scratch on 16x16 | 64 coarse traj | 0.00918±0.00052 | 0.00873±0.00049 | 0.0102±0.000403 | 0.0512±0.00412 | 0.0623±0.00218 | 1.0e-06 | 3.8e-05 | 0/64 |
| (c) GNN-ODE trained from scratch on 16x16 | 64 coarse traj | 0.0712±0.00977 | 0.106±0.0211 | 0.219±0.00695 | 0.505±0.017 | 0.21±0.0175 | 8.9e-01 | -1.1e+01 | 50/64 |
| (c) TC-ME trained from scratch on 8x8 (vs 8x8 reference) | 64 coarse traj | 0.0453±0.000729 | 0.0413±0.00028 | 0.0431±0.000127 | 0.195±0.00447 | 0.177±0.00101 | 1.7e-06 | 9.4e-05 | 0/64 |
| (c) Fisher-Hodge trained from scratch on 8x8 (vs 8x8 reference) | 64 coarse traj | 0.0513±0.000821 | 0.0503±0.00182 | 0.0485±0.00121 | 0.202±0.00091 | 0.199±0.00185 | 1.5e-06 | 7.3e-05 | 0/64 |
| (c) Flux-GNN trained from scratch on 8x8 (vs 8x8 reference) | 64 coarse traj | 0.028±0.00135 | 0.0297±0.00178 | 0.0374±0.000157 | 0.184±0.00895 | 0.157±0.00762 | 3.0e-06 | 2.8e-04 | 0/64 |

## Cross-scale readout consistency, test_id

pi err = ||pibar - P pi_fine|| / ||P pi_fine|| with pi_fine the stationary state of the 32x32 model of the same class (RG and scratch TC-ME: the fine TC-ME). J err = same for the stationary edge flux per unit length (P = flux-consistent coarse graining). "true" = vs the block-averaged 64x64 stationary reference. sigma = model EPR in probability units at its own stationary state (sigma_32 = fine model of the same class). mono = #trajectories (of 32 = 16 x 2 seeds) with sigma_32 >= sigma_16 >= sigma_8.

| model | pi err vs P pi_fine (16) | pi err vs true (16) | J err vs P J_fine (16) | J err vs true (16) | sigma_32 | sigma_16 | sigma_8 | sigma_16/sigma_32 | sigma_8/sigma_16 | mono |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) RG-TC-ME | 1.12e-15±5.56e-17 | 0.00314±0.000121 | 7.6e-16±6.06e-18 | 0.00896±0.000229 | 1.57±0.00043 | 0.754±0.000219 | 0.333±2.48e-05 | 0.484±2.12e-05 | 0.443±4.42e-05 | 32/32 |
| (c) scratch TC-ME | 0.0104±2.34e-05 | 0.0127±8.93e-06 | 0.0284±0.000687 | 0.0367±0.000768 | 1.57±0.00043 | 1.59±0.000758 | 1.42±0.00956 | 1.02±0.00164 | 0.905±0.00415 | 5/32 |
| (c) scratch Fisher-Hodge | 0.0107±8.61e-05 | 0.0132±0.000126 | 0.0355±0.000227 | 0.0449±0.000854 | 1.57±0.00554 | 1.56±0.000676 | 1.63±0.0157 | 1.01±0.00198 | 1.04±0.0143 | 8/32 |
| (c) scratch Flux-GNN (sigma formal) | 0.00814±0.000156 | 0.0103±0.000444 | 0.0212±3.05e-05 | 0.0257±0.000326 | 1.55±0.00116 | 1.49±0.00266 | 1.4±0.0383 | 0.969±0.00295 | 0.92±0.0282 | 29/32 |
| (c) scratch GNN-ODE | n/a | 0.231±0.00348 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| reference: sum j^2/(D rho) h^2 of the block-averaged true stationary state | - | - | - | - | 1.54 | 1.48 | 1.32 | 0.964 | 0.883 | (sigma_64 = 1.56) |

## Cross-scale readout consistency, test_oodpot

pi err = ||pibar - P pi_fine|| / ||P pi_fine|| with pi_fine the stationary state of the 32x32 model of the same class (RG and scratch TC-ME: the fine TC-ME). J err = same for the stationary edge flux per unit length (P = flux-consistent coarse graining). "true" = vs the block-averaged 64x64 stationary reference. sigma = model EPR in probability units at its own stationary state (sigma_32 = fine model of the same class). mono = #trajectories (of 32 = 16 x 2 seeds) with sigma_32 >= sigma_16 >= sigma_8.

| model | pi err vs P pi_fine (16) | pi err vs true (16) | J err vs P J_fine (16) | J err vs true (16) | sigma_32 | sigma_16 | sigma_8 | sigma_16/sigma_32 | sigma_8/sigma_16 | mono |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) RG-TC-ME | 1.58e-15±4.04e-17 | 0.0156±5.95e-05 | 5.32e-16±2.95e-17 | 0.0254±0.00166 | 6.41±0.0345 | 2.97±0.00967 | 1.17±0.0038 | 0.466±5.2e-05 | 0.396±8.13e-05 | 32/32 |
| (c) scratch TC-ME | 0.053±0.000297 | 0.0642±0.000183 | 0.063±0.000157 | 0.083±0.00237 | 6.41±0.0345 | 6.06±0.0165 | 4.33±0.0381 | 0.964±0.00133 | 0.793±0.000836 | 24/32 |
| (c) scratch Fisher-Hodge | 0.044±0.000296 | 0.0537±0.00145 | 0.0733±0.000625 | 0.0935±0.00134 | 6.54±0.0163 | 6.2±0.00243 | 5.75±0.105 | 0.964±0.00256 | 0.989±0.0194 | 13/32 |
| (c) scratch Flux-GNN (sigma formal) | 0.0392±0.000232 | 0.0507±0.0043 | 0.0685±0.00345 | 0.0904±0.00122 | 5.97±0.0697 | 5.46±0.256 | 4.34±0.165 | 0.95±0.0546 | 0.79±0.0222 | 22/32 |
| (c) scratch GNN-ODE | n/a | 0.529±0.0246 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| reference: sum j^2/(D rho) h^2 of the block-averaged true stationary state | - | - | - | - | 6.31 | 5.77 | 4.59 | 0.927 | 0.782 | (sigma_64 = 6.49) |

## Assertions on the closed-form rule (float64; 2 seeds x 2 sets x 16 trajectories)

| property | worst case | status |
|---|---|---|
| P pi stationary for Qbar: max|d^T Jbar| / max(pi_a escape_a) | 1.5e-15 | pass (<1e-9) |
| dense stationary vector of Qbar vs P pi (rel L2) | 9.0e-15 | pass |
| stationary flux exact: Jbar vs sum of crossing fine fluxes (16 and 8) | 5.9e-15 | pass |
| sigma_16 <= sigma_32 (max ratio) | 0.4918 | pass, all 64 |
| sigma_8 <= sigma_16 (max ratio) | 0.4691 | pass, all 64 |
| semigroup RG_4 = RG_2 o RG_2 | 1.8e-16 | pass |
| (kbar, phibar, Abar) reconstructs Qbar; kbar symmetric | 1.0e-16 | pass |
| A := 0 in fine model => Abar = 0 with phibar = lse(phi): max|Abar| | 2.1e-07 (fine-level float32 floor 4.0e-07) | pass (at float32 floor) |
| Abar in cycle space? Euclidean gradient fraction of Abar (median / max) | 0.085 / 0.178 (fine A: 4e-16) | NO: lse phibar is not the Hodge potential of the coarse force |
| kbar / (D n_16^2) (median) vs fine k / (D n_32^2) | 1.988 vs 0.997 | projected generator hops 2x faster (D_eff = m D) |
| g rho-sensitivity: max|dphi/dlog rho| / max|d phi| | 5.30e-02 | closure vs frozen differ little |

## Reading

* **Steady-state readouts: exact, as predicted.** RG-TC-ME (zero coarse data) reproduces P pi_fine and P J_fine to 1e-15 on
  16x16 (and 8x8), so its coarse readouts inherit the fine model's accuracy vs truth (pi 0.0031 / J 0.009 ID, vs 0.010-0.013 /
  0.026-0.045 for every model trained from scratch on 16x16). Scratch models disagree with P pi_fine by ~1% (ID) / 4-5% (OOD) and with
  P J_fine by 2-4% / 6-7%: a coarse model learned from data is a different object from the coarse-grained fine model.
* **EPR: chain monotone for RG (64/64 trajectories, both steps); scratch models are not ordered** (TC-ME 5/32 ID, FH 8/32, formal
  Flux-GNN 29/32 ID). But the RG sigma is a strict *lower bound* that drops by ~2x per level (sigma_16/sigma_32 = 0.48, sigma_8/sigma_16
  = 0.44 ID): the intra-block edges and the log-sum loss remove ~half the dissipation per level. The scratch models instead track the
  "physical" discretised EPR sum j^2/(D rho) h^2 of the block-averaged truth (ratio 0.96 / 0.88), which is NOT a coarse-graining-monotone
  quantity for a single Markov chain but is what a coarse model fitted to data reproduces. So "sigma ordered" holds only in the
  projection sense (hidden EPR of Esposito), not as resolution-independence of the learned EPR.
* **Transient dynamics: the projected generator is a poor propagator.** The 2x2 block chain is not lumpable; with the fixed in-block
  weights pi_i/pi_a every coarse hop rate is ~2x the natural D/h_c^2 (kbar/(D n^2) = 1.99, D_eff = m D in the diffusive limit), and the
  stationary flux is exact only because the in-block profile is frozen at pi. Hence 10T error = fine-model level (0.0030 ID / 0.0154 OOD,
  4x better than scratch TC-ME 0.0127 / 0.0634, near the fine-then-lump bound (b)), but at t = T, 2T the RG model relaxes too fast
  (0.305 / 0.164 ID vs 0.014 / 0.012 scratch). Halving kbar (diagnostic only; it breaks flux exactness by exactly 2x) gives 0.085 at T.
  A transient-accurate closed-form coarse model needs a non-Markov / memory correction (Mori-Zwanzig) or a time-rescaled rate with a
  matching drift correction; not done here.
* **Parametric form:** kbar symmetric, (kbar, phibar = lse phi, Abar) reconstructs Qbar exactly; detailed balance is preserved
  (A = 0 => Abar = 0 at the float32 floor). Under drive, Abar is not purely cycle-space (Euclidean gradient fraction 8.5% median,
  max 18%): lse(phi) is not the Hodge potential of the coarse force, so the coarse TC-ME needs a small gradient part in A (or a
  re-fitted phibar) to be in the canonical gauge.
* g depends on rho (max |d phi/d log rho| / max|d phi| = 5%); local-equilibrium closure and frozen generator give the same
  numbers to 3 digits.
* **Reverse direction (not run):** a TC-ME trained only on 16x16 data cannot produce a 32x32 model: P is many-to-one (sub-block
  profiles pi_i/pi_a, intra-block rates, and the dissipation on intra-block edges are absent from coarse data; sigma_16 <= sigma_32
  is a strict inequality with no inverse). One can evaluate the coarse-trained TC-ME on a 32x32 grid (it is resolution-agnostic),
  but that is extrapolation of the functional form, not inversion of the RG map.
* GNN-ODE (unstructured, no flux / no EPR): 10T error 0.22 ID, mass drift up to 0.89, negative density in 50/64 trajectories.

![EPR hierarchy](epr_hierarchy.png)

![steady flux match](steady_flux_match.png)

## E4b: lag-tau propagator coarse-graining (Markov-state-model style)

Kbar(tau) = P exp(tau Q_fine) Pi (Pi_ia = pi_i/pi_a, P = block sum), Q_fine = trained 32x32 TC-ME generator at its stationary state. Coarse rollout rhobar_{n+1} = Kbar(tau)^T rhobar_n (tau = 0.2: one step = 4 DT). Lbar = logm(Kbar)/tau; "NN-TC-ME" = Lbar truncated to grid edges (non-NN entries dropped, negatives clipped), written in TC-ME form (kbar, phibar = lse phi, Abar) and rolled out with exp(DT Lbar_NN). sigma_ref,16 = sum j^2/(D rho) h^2 of the block-averaged true stationary state (same discrete formula on 16x16). Median over 16 traj, mean±std over 2 seeds.

| coarse model (16x16) | T ID | 2T ID | 10T ID | T OOD | 2T OOD | 10T OOD |
|---|---|---|---|---|---|---|
| (a) projected generator (E4) | 0.304±8.41e-05 | 0.164±2.79e-05 | 0.00298±8.83e-05 | 0.267±0.000332 | 0.14±7.69e-05 | 0.0154±0.000381 |
| lag propagator Kbar(tau=0.05) | 0.104±7.87e-05 | 0.0659±4.16e-05 | 0.0029±1.36e-05 | 0.0806±0.000158 | 0.0511±2.57e-05 | 0.0152±0.000367 |
| lag propagator Kbar(tau=0.2) | 0.0282±6.92e-05 | 0.019±0.000138 | 0.00288±7.61e-05 | 0.0259±0.000166 | 0.0225±0.000418 | 0.0153±0.000165 |
| NN-TC-ME from logm Kbar(tau=0.05) | 0.297±5.34e-05 | 0.163±3.14e-05 | 0.00542±0.000122 | 0.329±0.000271 | 0.158±0.000179 | 0.0301±0.000138 |
| NN-TC-ME from logm Kbar(tau=0.2) | 0.286±1.24e-05 | 0.152±0.000109 | 0.0104±0.000152 | 0.308±0.0112 | 0.203±0.0139 | 0.0948±0.00485 |
| (b) fine TC-ME rollout then lump (E2) | 0.00329±7.05e-05 | 0.00291±1.21e-05 | 0.00309±0.000124 | 0.0128±0.000179 | 0.0129±6.29e-05 | 0.0154±6.48e-05 |
| (c) TC-ME trained on 16x16 | 0.0139±4.65e-05 | 0.0121±1.71e-05 | 0.0127±6.45e-06 | 0.0597±4.11e-05 | 0.0652±0.000715 | 0.0634±0.000155 |
| (c) Fisher-Hodge trained on 16x16 | 0.0152±5.48e-05 | 0.0137±0.000162 | 0.0132±0.000148 | 0.0602±0.000484 | 0.0607±0.000253 | 0.0517±0.000825 |
| (c) Flux-GNN trained on 16x16 | 0.00918±0.00052 | 0.00873±0.00049 | 0.0102±0.000403 | 0.0623±0.00218 | 0.0604±0.00349 | 0.0512±0.00412 |

| check (worst over 2 seeds x 2 sets x 16 traj) | tau = 0.05 | tau = 0.2 |
|---|---|---|
| Ppi stationary for Kbar (max |Ppi Kbar - Ppi| / max Ppi) | 4.1e-15 | 1.7e-14 |
| row sums - 1; min entry of Kbar | 7.1e-15; 2.7e-27 | 2.5e-14; 1.3e-13 |
| lag-tau block flux == fine lag-tau flux (rel. max) | 1.7e-14 | 1.1e-14 |
| logm: most negative off-diagonal / max NN rate | -5.26e-01 | -5.45e+00 |
| logm: # negative off-diagonal entries (max per traj, of 65280) | 32473 | 32503 |
| logm: share of |off-diagonal| mass on non-NN pairs (median) | 0.212 | 0.627 |
| logm: min NN rate / max NN rate | 6.02e-03 | -6.40e-01 |
| logm: max |imag|; Ppi Lbar = 0 residual | 0.0e+00; 8.8e-15 | 1.7e+01; 8.7e-15 |
| NN-TC-ME: Ppi stationary residual (max) | 1.0e-02 | 5.3e-02 |
| NN-TC-ME: stationary pi vs Ppi (median rel L2) | 1.13e-02 | 5.01e-02 |
| NN-TC-ME: Jbar(Ppi) vs P J_fine (median rel L2) | 4.85e-01 | 6.27e-01 |
| NN-TC-ME: kbar / (D n_16^2) (median) | 1.930 | 1.798 |
| NN-TC-ME: (k, phi, A) reconstructs rates | 2.4e-16 | 2.1e-14 |

| EPR / sigma_ref,16 (median; ID / OOD) | value |
|---|---|
| fine TC-ME sigma_32 / sigma_ref,16 | 1.06±0.000505 / 1.11±0.00209 |
| projected generator sigma_16 / sigma_ref,16 (E4) | 0.512±0.00095 / 0.518±0.00196 |
| lag-0.05 path EPR rate (1/tau) sum pi_a Kbar_ab log(pi_a Kbar_ab / pi_b Kbar_ba) / sigma_ref,16 | 0.766±0.000159 / 0.674±0.00217 |
| NN-TC-ME (tau=0.05) Schnakenberg sigma at Ppi / sigma_ref,16 | 1.17±0.00136 / 1.14±0.00575 |
| lag-0.2 path EPR rate (1/tau) sum pi_a Kbar_ab log(pi_a Kbar_ab / pi_b Kbar_ba) / sigma_ref,16 | 0.677±0.00144 / 0.381±0.000404 |
| NN-TC-ME (tau=0.2) Schnakenberg sigma at Ppi / sigma_ref,16 | 1.46±0.000758 / 11.2±0.673 |
| scratch TC-ME sigma_16 / sigma_ref,16 (test_id) | 1.08±0.00193 |
| scratch TC-ME sigma_16 / sigma_ref,16 (test_oodpot) | 1.07±0.007 |

![E4b](E4b_transient.png)
### E4b reading

* The lag propagator keeps everything exact that it should: Ppi stationary (1e-14), Kbar row-stochastic and non-negative, lag-tau block flux = fine lag-tau flux (1e-14), and 10T error stays at the fine-model level (0.0029 ID / 0.015 OOD).
* Transients improve with tau: error at t = T goes 0.30 (generator) -> 0.10 (tau = 0.05) -> 0.028 (tau = 0.2) on ID, and 0.27 -> 0.081 -> 0.026 on OOD. At tau = 0.2 it beats the 16x16-trained TC-ME on OOD at every time (0.026 / 0.023 / 0.015 vs 0.060 / 0.065 / 0.063) and on ID at 10T, but on ID at T and 2T it is still ~2x worse (0.028 / 0.019 vs 0.014 / 0.012). Outputs exist only on the tau grid.
* logm(Kbar)/tau is NOT a legal generator: about half the off-diagonal entries are negative (most negative = -0.53x the largest NN rate at tau = 0.05; -5.4x at tau = 0.2, which also has complex logm for OOD), and 21% / 63% of the off-diagonal mass sits on non-neighbour pairs. Forcing it into an NN TC-ME (truncate + clip) loses exactness (pi err 1-5%, J err 48-63%) and does not fix transients (T error 0.29-0.33). So there is no TC-ME *generator* (k, phi, A) behind the lag-tau coarse model; the improvement lives in the discrete-time propagator only.
* EPR against the fair baseline sigma_ref,16 (block-averaged truth, same discrete formula): projected generator 0.51 (ID) / 0.52 (OOD); lag-0.05 path EPR rate 0.77 / 0.67; lag-0.2 0.68 / 0.38; scratch TC-ME 1.08 / 1.07; fine TC-ME sigma_32 1.06 / 1.11. Every Markov coarse-graining stays a lower bound and falls short of sigma_ref,16.
* Verdict: in discrete time, the lag-tau propagator gets usable transients (at tau = 0.2) and exact steady states together. It does not give a legal continuous-time TC-ME generator, and its EPR is still a lower bound (0.4-0.8 of sigma_ref,16).
