#!/usr/bin/env bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 "$(dirname "$dir")"/data_generator.py -N 1024 -M 1 -s 100