"""All the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper_results_replication").resolve()
DOCS = SRC.joinpath("..", "..", "docs").resolve()

__all__ = ["BLD", "SRC", "TEST_DIR", "PAPER_DIR", "DOCS"]
