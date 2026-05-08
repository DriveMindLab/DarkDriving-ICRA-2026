#!/usr/bin/env sh
set -eu

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

OPT="${OPT:-options/test/darkdriving_lle.yml}"
SAVE_DIR="${SAVE_DIR:-results/darkdriving_release}"
"$PYTHON" test_darkdriving.py -opt "$OPT" --save_dir "$SAVE_DIR"
