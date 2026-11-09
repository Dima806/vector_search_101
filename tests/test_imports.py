"""Guard rails: "from scratch" stays true, and importing src stays light."""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
COMPILED = re.compile(r"^\s*(import|from)\s+(faiss|hnswlib)\b", re.MULTILINE)
HEAVY_TOP_LEVEL = re.compile(r"^(import|from)\s+(torch|sentence_transformers)\b", re.MULTILINE)


@pytest.mark.parametrize("path", sorted((SRC / "indexes").glob("*.py")), ids=lambda p: p.name)
def test_indexes_do_not_import_compiled_libraries(path: Path) -> None:
    assert not COMPILED.search(path.read_text()), f"{path.name} imports faiss/hnswlib"


def test_compiled_libraries_only_in_reference() -> None:
    offenders = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if COMPILED.search(p.read_text()) and p.parent.name != "reference"
    ]
    assert offenders == []


def test_no_top_level_torch_import() -> None:
    offenders = [p.name for p in SRC.rglob("*.py") if HEAVY_TOP_LEVEL.search(p.read_text())]
    assert offenders == []


def test_no_gpu_code() -> None:
    gpu = re.compile(r"\.cuda\(|device\s*=\s*[\"']cuda")
    assert not [p.name for p in SRC.rglob("*.py") if gpu.search(p.read_text())]
