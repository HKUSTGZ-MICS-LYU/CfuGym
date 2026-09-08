"""Shared helpers for the CFU agent tests."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_files(path: str) -> Path:
    return REPO_ROOT / path
