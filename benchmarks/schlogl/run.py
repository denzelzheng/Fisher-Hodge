"""Entry point of the schlogl benchmark; see benchmarks/reaction_networks/cli.py for the commands.

    python -m benchmarks.schlogl.run generate
    python -m benchmarks.schlogl.run train --model chart
    python -m benchmarks.schlogl.run tables
    python -m benchmarks.schlogl.run gifs
"""
from benchmarks.reaction_networks.cli import main

if __name__ == '__main__':
    main('schlogl')
