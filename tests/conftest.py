import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def adbulla_ocr() -> str:
    return (FIXTURES / "adbulla_ocr.txt").read_text(encoding="utf-8")


@pytest.fixture
def akshay_ocr() -> str:
    return (FIXTURES / "akshay_ocr.txt").read_text(encoding="utf-8")


@pytest.fixture
def amy_ocr() -> str:
    return (FIXTURES / "amy_ocr.txt").read_text(encoding="utf-8")


@pytest.fixture
def adbulla_input() -> Path:
    return ROOT / "input_data" / "Adbulla"


@pytest.fixture
def amy_input() -> Path:
    return ROOT / "input_data" / "Amy"


@pytest.fixture
def akshay_input() -> Path:
    return ROOT / "input_data" / "Akshay"
