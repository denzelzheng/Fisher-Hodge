"""Train one model on one benchmark.

    python -m scripts.train --benchmark porous_media --model chart --fast --device cuda
"""
from scripts._dispatch import run

if __name__ == '__main__':
    run('train', __doc__)
