#!/bin/sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

jupytext --to py $DIR/notebooks/*.ipynb
mv $DIR/notebooks/*.py src/
