#!/usr/bin/env bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 "$(dirname "$dir")"/calculate_optimal_T.py -N 1024 -M 10 --period 10 --verbose -s 100