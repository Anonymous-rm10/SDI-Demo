"""Launch the bundled train-jev service in its own environment."""
import os
import runpy
import sys

from .config import load_env
from .paths import project_path, use_source


def arguments(extra=()):
    return ["--base", str(project_path("JEV_BASE")),
            "--adapter", str(project_path("JEV_ADAPTER")),
            "--host", os.getenv("JEV_HOST", "127.0.0.1"),
            "--port", os.getenv("JEV_PORT", "8020"), *extra]


def main():
    load_env()
    use_source("JEV_REPO")
    sys.argv = ["trainjev.serve", *arguments(sys.argv[1:])]
    runpy.run_module("trainjev.serve", run_name="__main__")


if __name__ == "__main__":
    main()
