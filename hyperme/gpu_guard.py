"""Use a GPU only when no other user has a process on it.

    python -m hyperme.gpu_guard            # print the indices of GPUs with no foreign processes (empty: none)
    python -m hyperme.gpu_guard --wait     # block until one is free, then print its index
"""
import argparse
import getpass
import subprocess
import time


def _q(args):
    return subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, check=True).stdout.strip().splitlines()


def foreign_pids(me=None):
    """{gpu index: [pids of processes owned by another user]}"""
    me = me or getpass.getuser()
    idx = {u.strip(): int(i) for i, u in (l.split(',') for l in _q(['--query-gpu=index,uuid', '--format=csv,noheader']))}
    out = {i: [] for i in idx.values()}
    for line in _q(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']):
        if not line.strip():
            continue
        uuid, pid = (x.strip() for x in line.split(','))
        user = subprocess.run(['ps', '-o', 'user=', '-p', pid], capture_output=True, text=True).stdout.strip()
        if user != me:                                    # unknown owner (process in another namespace) counts as foreign
            out[idx[uuid]].append(int(pid))
    return out


def free_gpus(me=None):
    return [i for i, p in sorted(foreign_pids(me).items()) if not p]


def is_free(index, me=None):
    return not foreign_pids(me).get(index, [None])


def require_free(device):
    """abort before touching a GPU that another user is using (device 'cuda' = the first visible GPU)."""
    import os
    if not str(device).startswith('cuda'):
        return
    phys = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    if not is_free(phys):
        raise SystemExit(f'GPU {phys} has processes of other users; wait with python -m hyperme.gpu_guard --wait')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait', action='store_true'); p.add_argument('--poll', type=float, default=300.0)
    a = p.parse_args()
    while True:
        f = free_gpus()
        if f or not a.wait:
            print(' '.join(map(str, f)))
            break
        time.sleep(a.poll)
