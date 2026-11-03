"""Render the comparison animations of assets/ from the shipped Neural Chart checkpoints and trained baselines.

    python -m scripts.make_gifs --benchmark brusselator --device cuda --weights runs/brusselator
    python -m scripts.make_gifs --benchmark schlogl --track      # assets/schlogl_track_<test set>.gif

--track renders the tracer animation instead: tracers follow the probability velocity v = J / rho of the edge
(hyperedge) flux of each model and turn magenta where the local entropy production is negative or a flux runs against
the physical force.
"""
from scripts._dispatch import run

if __name__ == '__main__':
    run('gifs', __doc__)
