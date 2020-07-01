#!/usr/bin/env python

import pandas as pd
import dvc.api
import s3fs
import os
import requests
import argparse
from time import sleep
from pprint import pprint

DATA_VERSION = 'v1.0'

class Runner(object):
    def __init__(self, env, verbose, count):

        if env == 'dev':
            host = 'http://localhost:5000'

        elif env == 'stage':
            host = os.environ['SELDON_STAGE_URL']
           
        elif env == 'prod':
            host = os.environ['SELDON_PROD_URL']
            
        self.predict_url = '{}/api/v1.0/predictions'.format(host)
        self.feedback_url = '{}/api/v1.0/feedback'.format(host)

        print('Prediction URL: {}'.format(self.predict_url))
        
        self.verbose = verbose
        self.count = count

    def load_data(self, path):
        resource_url = dvc.api.get_url(
                        path=path,
                        repo=os.environ['DATA_REPO'],
                        rev=DATA_VERSION)

        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': os.environ['S3_ENDPOINT_URL']})
        return pd.read_csv(fs.open(resource_url))

    def send_request(self, payload):
        return requests.post(self.predict_url, json = payload).json()
                            
    def send_feedback(self, payload, truth, reward):
        # Truth. Given by the model 
        # Reward -> User says model is correct

        data = { 'response': payload,         
                'reward': reward, 
                'truth': {'data': {'ndarray': [truth]}} 
                }                         
        return requests.post(self.feedback_url, json = data).json()
        
    def run(self):
        nofraud_df = self.load_data('creditcard-nofraud.csv')
        fraud_df = self.load_data('creditcard-fraud.csv')

        THRESHOLD = 0.7

        print('Running test for {} transactions'.format(self.count))
        for _ in range(self.count):
            response = self.send_request({'data': {'ndarray': nofraud_df.sample(1).values.tolist()}})
            proba_nofraud = response['data']['ndarray'][0]    
            self.send_feedback(response, proba_nofraud < THRESHOLD, 0) 
        
            response = self.send_request({'data': {'ndarray': fraud_df.sample(1).values.tolist()}})
            proba_fraud = response['data']['ndarray'][0]
            self.send_feedback(response, proba_fraud > THRESHOLD, 1) 

            if self.verbose:
                print("Sending Class='0'. Result: {:.4f}".format(proba_nofraud))
                print("Sending Class='1'. Result: {:.4f}".format(proba_fraud))

            sleep(0.1)

        print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help="environment")
    parser.add_argument("-v", "--verbose", help="verbose", action='store_true')
    parser.add_argument("-c", "--count", help="count", default=10)
    args = parser.parse_args()

    r = Runner(args.environment, args.verbose, int(args.count))
    r.run()
