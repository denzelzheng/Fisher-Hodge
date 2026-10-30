"""Write the tables of results/<benchmark>/ (markdown and csv) from the run files in results/<benchmark>/runs/.

    python -m scripts.make_tables --benchmark all
    python -m scripts.make_tables --benchmark schlogl --runs runs/schlogl     # tables from your own runs
"""
from scripts._dispatch import run

if __name__ == '__main__':
    run('tables', __doc__)
