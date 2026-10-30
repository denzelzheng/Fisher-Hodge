"""Compact physics-consistency table of one benchmark: for every model the largest value over the test sets of
(edges violating local detailed balance, samples with sigma < 0, negative values), read from results/<name>/physics.csv.
These are the physics tables of the README.

    python -m scripts.physics_summary --benchmark porous_media
"""
import argparse
import csv
import math
import os

from benchmarks.common import RESULTS, fmt
from scripts._dispatch import BENCHMARKS


def summary(name, models=None):
    rows = list(csv.reader(open(os.path.join(RESULTS, name, 'physics.csv'))))
    out, order = {}, []
    for r in rows[1:]:
        m, q, vals = r[0], r[1].lower(), r[2:]
        if m not in out:
            out[m] = {}
            order.append(m)
        nums = [float(v) for v in vals if v not in ('', 'N/A')]
        key = 'ldb' if 'detailed balance' in q else 'sigma' if 'sigma' in q or 'σ' in q else 'neg'
        out[m][key] = max(nums) if nums else None
    lines = ['| Method | edges violating local detailed balance | samples with σ < 0 | negative values |', '|---|---|---|---|']
    for m in (models or order):
        if m in out:
            d = out[m]
            cell = lambda k: 'N/A' if d.get(k) is None else fmt(d[k]) if not (isinstance(d[k], float) and math.isnan(d[k])) else 'N/A'
            lines.append(f'| {m} | {cell("ldb")} | {cell("sigma")} | {cell("neg")} |')
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=BENCHMARKS)
    print(summary(p.parse_args().benchmark))
