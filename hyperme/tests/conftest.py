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
    return 'cuda' if torch.cuda.is_available() else 'cpu'
