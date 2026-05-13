import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    if os.environ.get("DISABLE_GPU_TESTS") == "1":
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def model_id() -> str:
    return "google/ddpm-celebahq-256"
