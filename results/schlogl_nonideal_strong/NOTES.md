# Notes

The non-ideal Schlögl model, ln γ(x) = 0.25 x / (1 − 0.25 x), with the chemostat a over 1.5–4.5 instead of 2.85–3.10. Concentrations reach 2.23 in training, where ln γ = 1.26.

## Data and test sets

- Training: 48 trajectories of 100 steps, a 1.5–4.5, b 1.0–3.2.
- Training conditions: 16 trajectories of 200 steps, same ranges.
- Extrapolation: 16 trajectories of 200 steps, a 4.6–5.0, b 2.0–3.5. Concentrations reach 2.30 and stay bounded.
- 16×16 grid: 8 trajectories of 200 steps, training ranges, same spacing, no retraining.

## Integrator tolerance

`calibrate` gives the oracle a median 10T error of 8.5e-7 and 1.0e-7 at every rtol from 1e-2 to 1e-5. The largest error over time and trajectories is 7.6e-5 at rtol 1e-4. The benchmark uses rtol 1e-4, as the other reaction networks do.
