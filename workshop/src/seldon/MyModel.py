import os
import logging
import joblib

logger = logging.getLogger(__name__)

class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """

        logger.info('Starting {} Microservice'.format(__name__))

        model_path = os.environ['HOME'] + '/model'
        self.model = joblob.load(model_path)
        self.model_metadata = {}
        
        self.cm = {'tp': 0.0, 'fp': 0.0, 'tn': 0.0, 'fn': 0.0}

        self.requests = 0
        self.feedback_given = 0
        self.accuracy = self.precision = self.recall = self.f1 = 0.0
        
    def predict(self, X, features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """

        logger.info('Predict function called')
        self.requests += 1
        return self.predict(X)

    def metadata(self):
        logger.info("metadata method")
        return {"metadata":{"modelName":"mean_classifier"}}

    def health_status(self):
        logger.info("health status method")
        return { "status": "ok" }

    def tags(self):
        logger.info("tags method")
        return self.model_metadata

    def metrics(self):
        logger.info('Model metrics method')

        tp = {"type": "GAUGE", "key": "true_pos_total",
              "value": self.cm['tp'], "tags": self.tags()}
        tn = {"type": "GAUGE", "key": "true_neg_total",
              "value": self.cm['tn'], "tags": self.tags()}
        fp = {"type": "GAUGE", "key": "false_pos_total",
              "value": self.cm['fp'], "tags": self.tags()}
        fn = {"type": "GAUGE", "key": "false_neg_total",
              "value": self.cm['fn'], "tags": self.tags()}

        accuracy = {"type": "GAUGE", "key": "branch_accuracy", "value": self.accuracy,
                   "tags": self.tags()}
        # success = {"type": "GAUGE", "key": "n_success_total", "value": self.success,
        #            "tags": self.tags()}
        requests = {"type": "GAUGE", "key": "n_requests_total", "value": self.requests,
                 "tags": self.tags()}
        feedback = {"type": "GAUGE", "key": "n_feedback_total", "value": self.feedback_given,
                 "tags": self.tags()}

        precision = {"type": "GAUGE", "key": "precision",
              "value": self.precision, "tags": self.tags()}
        recall = {"type": "GAUGE", "key": "recall",
              "value": self.recall, "tags": self.tags()}
        f1 = {"type": "GAUGE", "key": "f1",
              "value": self.f1, "tags": self.tags()}

        return [tp, tn, fp, fn, accuracy, requests, feedback, precision, recall, f1]

    def send_feedback(self, features, feature_names, reward, truth, routing):
        logger.info('Model send-feedback method')
        logger.info('Routing: {}'.format(routing))
        
        # Truth. Given by the model 
        # Reward -> User says model is correct

        # truth is a list
        truth = truth[0]
        logger.info(f"Reward: {reward} Truth: {truth}")

        if reward == 1:
            if truth == 1:
                self.cm['tp'] += 1
            elif truth == 0:
                self.cm['tn'] += 1
        elif reward == 0:
            if truth == 1:
                self.cm['fn'] += 1
            elif truth == 0:
                self.cm['fp'] += 1

        logger.info(self.cm)
        
        self.feedback_given+=1 
        #self.success = self.success + 1 if reward else self.success
        self.accuracy = (self.cm['tp'] + self.cm['tn']) / sum(self.cm.values())
        
        self.precision = self.cm['tp'] / (self.cm['tp'] + self.cm['fp']) if self.cm['tp'] + self.cm['fp'] > 0 else 0
        self.recall = self.cm['tp'] / (self.cm['tp'] + self.cm['fn']) if self.cm['tp'] + self.cm['fn'] > 0 else 0
        self.f1 = 2 * ((self.precision * self.recall)/(self.precision + self.recall)) if self.precision + self.recall > 0 else 0
        
        logger.info("Feedback given: {} Accuracy: {} Precision: {} Recall: {} F1: {}". \
                    format(self.feedback_given, self.accuracy, self.precision, self.recall, self.f1))
        