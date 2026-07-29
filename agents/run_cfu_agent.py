#!/usr/bin/env python3
"""Run the LangGraph CFU design agent from the repo root."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.cfu_design_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

