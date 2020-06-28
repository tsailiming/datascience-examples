#!/usr/bin/env/python

import dvc.api
import s3fs
import os

import requests
from time import sleep

DATA_VERSION = 'v1.0'

def load_data(path):
    resource_url = dvc.api.get_url(
                    path=path,
                    repo=os.environ['DATA_REPO'],
                    rev=DATA_VERSION)

    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': os.environ['S3_ENDPOINT_URL']})
    return pd.read_csv(fs.open(resource_url))

def send_request(payload):
    return requests.post('{}/api/v1.0/predictions'.format(os.environ['SELDON_PROD_URL']), json = payload).json()
                         
def send_feedback(payload, truth, reward):
    # Truth. Given by the model 
    # Reward -> User says model is correct

    data = { 'response': payload,         
            'reward': reward, #random.randint(0,1), 
            'truth': {'data': {'ndarray': [truth]}} 
            }                         
    return requests.post('{}/api/v1.0/feedback'.format(os.environ['SELDON_PROD_URL']), json = data).json()


nofraud_df = load_data('creditcard-nofraud.csv')
fraud_df = load_data('creditcard-fraud.csv')

THRESHOLD = 0.7
RANGE = 500

print('Running test for {} transactions'.format(RANGE))
for _ in range(RANGE):
    response = send_request({'data': {'ndarray': nofraud_df.sample(1).values.tolist()}})
    proba = response['data']['ndarray'][0]
    send_feedback(response, proba < THRESHOLD, 0) 
  
    response = send_request({'data': {'ndarray': fraud_df.sample(1).values.tolist()}})
    proba = response['data']['ndarray'][0]
    send_feedback(response, proba > THRESHOLD, 1) 

    sleep(0.1)

print('DONE')



