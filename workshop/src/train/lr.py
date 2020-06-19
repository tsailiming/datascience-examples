#!/usr/bin/env python

import os
import shutil
import argparse

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, recall_score, precision_score, \
    average_precision_score, precision_recall_curve, f1_score, auc, \
    roc_curve, roc_auc_score, confusion_matrix, accuracy_score

import pandas as pd
import numpy as np

DATA_VERSION = ''


class Run:
    def __init__(self, run_name, model_path):
        self.run_name = run_name
        self.model_path = model_path
        self.experiment_id = self._get_experiement_id()
        self._threshold = 0.5
        self._random_seed = 42
        self._test_size = 0.3

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        self._random_seed = random_seed

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, test_size):
        self._test_size = test_size

    def _get_experiement_id(self):
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'Dev')

        e = mlflow.get_experiment_by_name(experiment_name)
        if not e:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            experiment_id = e.experiment_id

        return experiment_id

    def _prepare_dataset(self):
        df = pd.read_csv('../../../creditcard.csv')

        features = df.columns.values

        # # Finding features with the highest correlation
        def most_corr(param, n):
            class_corr = df.corr()[param].sort_values(ascending=False)
            list_class = []
            for i in features:
                if(np.abs(class_corr[i]) >= n):
                    list_class.append(i)
            return list_class
        # Select features with correlation higher than 0.1 (positive correlation) or lower than -0.1 (negative correlation)
        selected_features = most_corr('Class', 0.1)

        dataset = df[selected_features]

        # ## Split the data into X and y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset.drop(
            'Class', 1), dataset['Class'], test_size=self.test_size, random_state=self.random_seed)

    def run(self):
        self._prepare_dataset()

        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name):

            mlflow.log_param('threshold', self.threshold)
            mlflow.log_param('random_seed', self.random_seed)
            mlflow.log_param('test_size', self.test_size)

            self.train()
            self.test()

    def train(self):

        resampling = SMOTE(sampling_strategy='minority',
                           random_state=self.random_seed)
        self.model = Pipeline(
            [('SMOTE', resampling), ('Logistic Regression', LogisticRegression())])

        self.model.fit(self.X_train, self.y_train)

        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)

        mlflow.sklearn.save_model(self.model, self.model_path)
        mlflow.sklearn.log_model(self.model, 'models')

    def test(self):
        # Probabilities
        y_proba_baseline = self.model.predict_proba(self.X_test)[:, 1]

        average_precision = average_precision_score(
            self.y_test, y_proba_baseline)
        mlflow.log_metric('average_precision', average_precision)

        rpt = classification_report(
            self.y_test, y_proba_baseline > self.threshold, output_dict=True)
        for lbl in ['0', '1']:
            mlflow.log_metric(lbl + '_recall', rpt[lbl]['recall'])
            mlflow.log_metric(lbl + '_f1_score', rpt[lbl]['f1-score'])
            mlflow.log_metric(lbl + '_precision', rpt[lbl]['precision'])

        accuracy = accuracy_score(
            self.y_test, y_proba_baseline > self.threshold)
        mlflow.log_metric('accuracy', accuracy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", help="experiment run name")
    parser.add_argument("-m", "--model-path",
                        help="path to the model", default='/tmp/model')
    args = parser.parse_args()

    r = Run(args.run_name, args.model_path)
    r.run()
