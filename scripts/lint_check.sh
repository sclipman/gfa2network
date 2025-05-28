#!/usr/bin/env bash
set -e
black --check gfa2network tests
flake8 gfa2network tests
