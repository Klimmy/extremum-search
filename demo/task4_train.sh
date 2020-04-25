#!/usr/bin/env bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 "$(dirname "$dir")"/model.py -N 1024 -M 100 --no-gpu --epochs 2 --learning-rate 0.001 --logging