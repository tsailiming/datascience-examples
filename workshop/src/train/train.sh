#!/bin/bash

pip install -U pip setuptools
pip install -r /workspace/source/workshop/src/train/requirements.txt

echo "Running training"
python /workspace/source/workshop/src/train/lr.py -m /workspace/model 