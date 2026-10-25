"""Use a GPU only when no other user has a process on it, except users we have agreed to share with.

    python -m hyperme.gpu_guard            # print usable GPU indices, least crowded first (empty: none)
    python -m hyperme.gpu_guard --wait     # block until one is usable, then print the indices
Sharing: HYPERME_GPU_SHARE_WITH=user1,user2 (or --share-with) lists the colleagues whose processes do not block a
GPU, e.g. yixunhu on machine 49.  Nobody else is ever shared with.
"""
import argparse
import getpass
import os
import subprocess
import time


def _q(args):
    return subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, check=True).stdout.strip().splitlines()


def share_list(share_with=None):
    s = share_with if share_with is not None else os.environ.get('HYPERME_GPU_SHARE_WITH', '')
    return {u.strip() for u in (s.split(',') if isinstance(s, str) else s) if u.strip()}


def gpu_processes(me=None, share_with=None):
    """{gpu index: (pids blocking it, number of processes on it)}; my own and allowed users' pids do not block."""
    ok = {me or getpass.getuser()} | share_list(share_with)
    idx = {u.strip(): int(i) for i, u in (l.split(',') for l in _q(['--query-gpu=index,uuid', '--format=csv,noheader']))}
    out = {i: ([], 0) for i in idx.values()}
    for line in _q(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']):
        if not line.strip():
            continue
        uuid, pid = (x.strip() for x in line.split(','))
        user = subprocess.run(['ps', '-o', 'user=', '-p', pid], capture_output=True, text=True).stdout.strip()
        block, n = out[idx[uuid]]
        if user not in ok:                                # unknown owner (process in another namespace) blocks
            block = block + [int(pid)]
        out[idx[uuid]] = (block, n + 1)
    return out


def free_gpus(me=None, share_with=None):
    g = gpu_processes(me, share_with)
    return [i for i, (b, n) in sorted(g.items(), key=lambda kv: (kv[1][1], kv[0])) if not b]


def is_free(index, me=None, share_with=None):
    return not gpu_processes(me, share_with).get(index, ([None], 0))[0]


def require_free(device):
    """abort before touching a GPU that a non-allowed user is using (device 'cuda' = the first visible GPU)."""
    if not str(device).startswith('cuda'):
        return
    phys = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    if not is_free(phys):
        raise SystemExit(f'GPU {phys} has processes of other users; wait with python -m hyperme.gpu_guard --wait')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait', action='store_true'); p.add_argument('--poll', type=float, default=300.0)
    p.add_argument('--share-with', default=None)
    a = p.parse_args()
    while True:
        f = free_gpus(share_with=a.share_with)
        if f or not a.wait:
            print(' '.join(map(str, f)))
            break
        time.sleep(a.poll)
