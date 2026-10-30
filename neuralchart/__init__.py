"""neuralchart: a thermodynamically consistent learned master equation on graphs and hypergraphs."""
from .graphs import Hypergraph, grid_edge_list
from .model import MasterEquation, FreeEnergy, Mobility, Drive, Context, MLP, featurizer
from .chart import ChartConfig, NeuralChart, CONFIGS, chart_config, fit_chart
from .fit import fit_linear, fit_varpro, train, rollout
from . import fast, integrators, diagnostics

__all__ = ['Hypergraph', 'grid_edge_list', 'MasterEquation', 'FreeEnergy', 'Mobility', 'Drive', 'Context', 'MLP',
           'featurizer', 'ChartConfig', 'NeuralChart', 'CONFIGS', 'chart_config', 'fit_chart', 'fit_linear',
           'fit_varpro', 'train', 'rollout', 'fast', 'integrators', 'diagnostics']
