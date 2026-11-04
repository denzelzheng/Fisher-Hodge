"""Entry point of the schlogl_nonideal_strong benchmark; see benchmarks/reaction_networks/cli.py for the commands.

    python -m benchmarks.schlogl_nonideal_strong.run generate
    python -m benchmarks.schlogl_nonideal_strong.run train --model chart
    python -m benchmarks.schlogl_nonideal_strong.run tables
    python -m benchmarks.schlogl_nonideal_strong.run gifs
"""
from benchmarks.reaction_networks.cli import main

if __name__ == '__main__':
    main('schlogl_nonideal_strong')
