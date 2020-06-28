#!/bin/bash

pip install -U pip setuptools gitdb==4.0.5 protobuf==3.12.2
pip install -r /projects/datascience-examples/workshop/src/train/requirements.txt

echo "Running testing"
python /projects/datascience-examples/workshop/src/prod/production-test.py