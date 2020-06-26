#!/usr/bin/env python

import pandas as pd
import numpy as np
import dvc.api
import s3fs
import os

import requests
from pprint import pprint

DATA_VERSION = 'v1.0'

def load_data(path):
    resource_url = dvc.api.get_url(
                    path=path,
                    repo=os.environ['DATA_REPO'],
                    rev=DATA_VERSION)

    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': os.environ['S3_ENDPOINT_URL']})
    return pd.read_csv(fs.open(resource_url))

def send_request(payload):
    return requests.post('http://localhost:5000/predict', json = payload).json()

nofraud_df = load_data('creditcard-nofraud.csv')
fraud_df = load_data('creditcard-fraud.csv')

print ("Sending Class='0'")
pprint(send_request({'data': {'ndarray': nofraud_df.sample(10).values.tolist()}})['data']['ndarray'])

print()

print ("Sending Class='1'")
pprint(send_request({'data': {'ndarray': fraud_df.sample(10).values.tolist()}})['data']['ndarray'])