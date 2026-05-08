#!/usr/bin/env sh
set -eu

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

OPT="${OPT:-options/train/darkdriving_lle.yml}"
"$PYTHON" train.py -opt "$OPT"
