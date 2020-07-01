#!/bin/bash

# Source code is checked ut at /workspace/source
# Model should be written to /workspace/model

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. $DIR/config.sh

pip install -U pip setuptools
pip install -r /workspace/source/workshop/src/train/requirements.txt

echo "Running training"
python /workspace/source/workshop/src/train/$PYTHON_SCRIPT -m /workspace/model -r $RUN_NAME
