"""Entry point of the phosphorylation_cycle benchmark; see benchmarks/reaction_networks/cli.py for the commands.

    python -m benchmarks.phosphorylation_cycle.run generate
    python -m benchmarks.phosphorylation_cycle.run train --model chart
    python -m benchmarks.phosphorylation_cycle.run tables
    python -m benchmarks.phosphorylation_cycle.run gifs
"""
from benchmarks.reaction_networks.cli import main

if __name__ == '__main__':
    main('phosphorylation_cycle')
