"""Entry point of the phosphorylation_heterogeneous benchmark; see benchmarks/reaction_networks/cli.py for the commands.

    python -m benchmarks.phosphorylation_heterogeneous.run generate
    python -m benchmarks.phosphorylation_heterogeneous.run train --model chart
    python -m benchmarks.phosphorylation_heterogeneous.run tables
    python -m benchmarks.phosphorylation_heterogeneous.run gifs
"""
from benchmarks.reaction_networks.cli import main

if __name__ == '__main__':
    main('phosphorylation_heterogeneous')
