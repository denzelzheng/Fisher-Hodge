# Third-party code and methods

This repository contains no third-party source code.
Two baselines are our own implementations of published methods.
One baseline needs the authors' code, which you install separately.

## HelmFluid (external code, installed separately)

- Paper: X. Xing, H. Wu, Y. Ma, J. Wang, M. Long, "HelmFluid: Learning Helmholtz Dynamics for Interpretable Fluid Prediction", ICML 2024.
- Code: https://github.com/thuml/HelmFluid, commit `d8003045425b89d4c2fdd84dced241a77495e50d` (2024-07-19). We used it unmodified.
- Licence: MIT.
- Use: clone the repository, install CuPy, and set `NEURALCHART_HELMFLUID` to the clone. `neuralchart/baselines/helmfluid.py` imports the model from there and adds static condition channels to its input. Its `forward` follows `models/HelmFluid_2D_corr_potential.py` of the upstream repository, so the MIT notice below applies to that function.

```
MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GENERIC-FNO (our implementation)

- Paper: J. Sulskis and R. Ravi, "GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators", arXiv:2606.08343.
- Authors' code: an anonymous repository of Jupyter notebooks, https://anonymous.4open.science/r/GENERIC-FNO-F414/ (fetched 2026-10-26). Its LICENSE file is the full text of the GNU General Public License v3.0.
- `neuralchart/baselines/generic_fno.py` implements the model from the paper's equations. Architecture details and hyperparameters that the paper leaves open follow the authors' public notebook. No code was copied. The file marks every such detail with `[notebook]`.
- The extensions for several fields and for condition channels are ours and are described in the file header.

## BSMS-GNN and MeshGraphNets (our implementation)

- Papers: Y. Cao, M. Chai, M. Li, C. Jiang, "Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network", ICML 2023. T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. Battaglia, "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021.
- `neuralchart/baselines/bsms_gnn.py` implements both from the papers. No code from the authors' repositories was used.

## Other baselines

FNO, Flux-GNN, the CRNN variants and the UDE closure are our own implementations of the published ideas (Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021; Ji and Deng, "Autonomous Discovery of Unknown Reaction Pathways from Data by Chemical Reaction Neural Network", J. Phys. Chem. A 2021; Rackauckas et al., "Universal Differential Equations for Scientific Machine Learning", 2020).
