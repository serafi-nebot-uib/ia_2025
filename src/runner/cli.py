# src/runner/cli.py

import typer
import subprocess
import sys
import os
from pathlib import Path

app = typer.Typer()

SRC_PATH = Path(__file__).resolve().parents[1]

def run_module(module: str):
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=SRC_PATH,
        env={**dict(PYTHONPATH=str(SRC_PATH)), **dict(**os.environ)},
    )

@app.command()
def aspirador():
    """Run the aspirador.main task."""
    run_module("aspirador")

@app.command()
def prova():
    print("Tot ha funcionat")

if __name__ == "__main__":
    app()
