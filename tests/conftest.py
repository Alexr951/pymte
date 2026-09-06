import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "oracle"


def load_oracle(name: str) -> dict:
    """Load a JSON fixture produced by tests/fixtures/run_oracle.R."""
    with (FIXTURES / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def oracle():
    return load_oracle
