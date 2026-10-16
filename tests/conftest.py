import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset          # noqa: E402
from src.features.preprocess import prepare       # noqa: E402
from src.utils.helpers import load_config         # noqa: E402

DEMO = ROOT / "data" / "synthetic_demo"


@pytest.fixture(scope="session")
def demo_config():
    return load_config()


@pytest.fixture(scope="session")
def demo_raw():
    return load_dataset(DEMO / "interactions.csv", DEMO / "users.csv", DEMO / "items.csv")


@pytest.fixture(scope="session")
def demo_data(demo_raw, demo_config):
    return prepare(demo_raw, demo_config)
