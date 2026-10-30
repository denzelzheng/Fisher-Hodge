"""Dispatch a command to the entry point of one benchmark."""
import argparse
import sys

BENCHMARKS = ('porous_media', 'porous_media_crowding', 'brusselator', 'schlogl', 'schlogl_nonideal',
              'phosphorylation_cycle', 'mapk_cascade')
REACTION_NETWORKS = ('schlogl', 'schlogl_nonideal', 'phosphorylation_cycle', 'mapk_cascade')


def run(command, doc):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog='Every other option is passed to python -m benchmarks.<name>.run; see its --help.')
    p.add_argument('--benchmark', required=True, choices=BENCHMARKS + ('all',))
    a, rest = p.parse_known_args()
    for name in (BENCHMARKS if a.benchmark == 'all' else (a.benchmark,)):
        if a.benchmark == 'all' and name == 'porous_media_crowding' and command == 'tables':
            continue                                  # the porous-media tables command writes both variants
        argv = [command] + rest
        if name.startswith('porous_media'):
            from benchmarks.porous_media.run import main
            main(argv + ['--variant', name])
        elif name == 'brusselator':
            from benchmarks.brusselator.run import main
            main(argv)
        else:
            from benchmarks.reaction_networks.cli import main
            main(name, argv)


if __name__ == '__main__':
    sys.exit('use scripts/train.py, evaluate.py, make_tables.py or make_gifs.py')
