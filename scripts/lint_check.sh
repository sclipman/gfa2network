#!/usr/bin/env bash
set -euo pipefail
python -m ruff check gfa2network tests scripts/benchmark.py
python -m ruff format --check gfa2network tests scripts/benchmark.py
