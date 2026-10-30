"""Evaluate one model on one benchmark.

    python -m scripts.evaluate --benchmark porous_media --model chart --fast --device cuda
"""
from scripts._dispatch import run

if __name__ == '__main__':
    run('evaluate', __doc__)
