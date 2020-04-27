#!/usr/bin/env bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 "$(dirname "$dir")"/model_evaluation.py -N 1024 -M 100 -k 3 -T 5.2 --no-gpu --threshold 0.5