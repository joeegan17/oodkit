#!/bin/bash

# Usage: ./run.sh path/to/script.py

if [ -z "$1" ]; then
  echo "Usage: ./run.sh path/to/script.py"
  exit 1
fi

docker compose run --rm dev python "$@"