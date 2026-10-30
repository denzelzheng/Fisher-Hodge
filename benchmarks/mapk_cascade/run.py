"""Entry point of the mapk_cascade benchmark; see benchmarks/reaction_networks/cli.py for the commands.

    python -m benchmarks.mapk_cascade.run generate
    python -m benchmarks.mapk_cascade.run train --model chart
    python -m benchmarks.mapk_cascade.run tables
    python -m benchmarks.mapk_cascade.run gifs
"""
from benchmarks.reaction_networks.cli import main

if __name__ == '__main__':
    main('mapk_cascade')
