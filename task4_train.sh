#!/usr/bin/env bash
python3 "$(dirname "$0")"/train_model.py -N 1024 -M 100 --no-gpu --epochs 2 --learning-rate 0.001 --logging