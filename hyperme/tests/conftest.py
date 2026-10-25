import os
import sys
import importlib.util
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault('FH_ROOT', REPO)


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: retrains models (minutes); deselect with -m "not slow"')


def load_legacy(name, relpath, extra_path=None):
    """import a prototype module by file path without letting it change global torch state."""
    if extra_path:
        p = os.path.join(REPO, extra_path)
        if p not in sys.path:
            sys.path.insert(0, p)
    dt = torch.get_default_dtype()
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.set_default_dtype(dt)
    return mod


@pytest.fixture(scope='session')
def device():
    """CPU unless HYPERME_DEVICE=cuda is set, and then only on a GPU without other users' processes."""
    dev = os.environ.get('HYPERME_DEVICE', 'cpu')
    if dev.startswith('cuda'):
        from hyperme.gpu_guard import is_free
        phys = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        if not torch.cuda.is_available() or not is_free(phys):
            pytest.skip(f'GPU {phys} is not free of other users; run hyperme/tests/run_gpu_regression.sh')
    return dev
