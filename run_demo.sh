#!/usr/bin/env bash
set -e
python run_demo.py --users 300 --output outputs --tuning quick --bootstrap 1000
