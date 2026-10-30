"""Baselines.  Own implementations: Flux-GNN and CRNN (graph_nets), BSMS-GNN (bsms_gnn), GENERIC-FNO (generic_fno).
HelmFluid is imported from a local clone of its repository (helmfluid, see THIRD_PARTY.md).  The benchmark-specific
wrappers (inputs, UDE closure, FNO, persistence) live in benchmarks/<name>/models.py."""
from .graph_nets import FluxGNN, CRNN, HyperGNNTrunk

__all__ = ['FluxGNN', 'CRNN', 'HyperGNNTrunk']
