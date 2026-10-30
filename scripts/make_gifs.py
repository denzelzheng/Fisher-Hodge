"""Render the comparison animations of assets/ from the shipped Neural Chart checkpoints and trained baselines.

    python -m scripts.make_gifs --benchmark brusselator --device cuda --weights runs/brusselator
"""
from scripts._dispatch import run

if __name__ == '__main__':
    run('gifs', __doc__)
