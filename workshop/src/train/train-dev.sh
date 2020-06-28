#!/bin/bash

pip install -U pip setuptools grpcio-tools==1.29.0 seldon-core==1.1.0 
pip install -r /projects/datascience-examples/workshop/src/train/requirements.txt

echo "Running training"
rm -rf $HOME/model
python /projects/datascience-examples/workshop/src/train/lr.py -m $HOME/model -r lr