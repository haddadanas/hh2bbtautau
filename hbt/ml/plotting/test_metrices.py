import sys
import unittest
import numpy as np
from metrices import *
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve


class Test_Conf_Matrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test = [np.random.randint(0,10*i, size=(1000*i,5)) for i in range(1, 10)]
        cls.trues = [np.random.randint(0,5,size= 1000*i) for i in range(1, 10)]
        cls.pred = [np.divide(test.T, np.sum(test, axis=1)).T for test in cls.test]
        cls.weights = [np.random.randint(0,5, size = 1000*i) for i in range(1, 10)]
    
    def test_unvalid_inputs_conf(self):
        for ind, (trues, pred) in enumerate(zip(self.trues, self.pred)):
            with self.assertRaises(ValueError):
                get_conf_matrix(np.array([1,0,1,1,0]), pred)
            with self.assertRaises(ValueError):
                get_conf_matrix(trues, pred[:100])
    
    def test_unvalid_weights_conf(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            with self.assertRaises(ValueError):
                get_conf_matrix(trues, pred, weights=weights[:100])
            with self.assertRaises(ValueError):
                get_conf_matrix(trues, pred, weights=np.stack((weights, weights), axis = 0))

    def test_equals_no_weights(self):
        for ind, (trues, pred) in enumerate(zip(self.trues, self.pred)):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1))
            my_matrix = get_conf_matrix(trues,pred)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())

    def test_equals_with_weights(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1),weights=weights)
            my_matrix = get_conf_matrix(trues,pred, weights=weights)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())
    
    def test_equals_with_errors(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1),weights=weights)
            my_matrix = get_conf_matrix(trues,pred, weights=weights, errors=True)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())
    
    def test_unvalid_inputs_roc(self):
        for ind, (trues, pred) in enumerate(zip(self.trues, self.pred)):
            trues = trues > 0
            pred = pred[:,0]
            with self.assertRaises(ValueError):
                get_roc_data(np.array([1,0,1,1,0]), pred)
            with self.assertRaises(ValueError):
                get_roc_data(trues, pred[:100])
            with self.assertRaises(ValueError):
                get_roc_data(trues, pred, (1-pred)[:10])
    
    def test_unvalid_weights_roc(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            trues = trues > 0
            pred = pred[:,0]
            with self.assertRaises(ValueError):
                get_roc_data(trues, pred, weights=weights[:100])
            with self.assertRaises(ValueError):
                get_roc_data(trues, pred, weights=np.stack((weights, weights), axis = 0))
  
    def test_equals_without_weights_roc(self):
        for ind, (trues, pred) in enumerate(zip(self.trues, self.pred)):
            trues = trues > 0
            pred = pred[:,0]
            sk_fpr, sk_tpr, sk_threshold = roc_curve(trues, pred)
            my_fpr, my_tpr, _  = get_roc_data(trues,pred,thresholds=sk_threshold, errors = False)
            self.assertTrue((sk_fpr == my_fpr).all())
            self.assertTrue((sk_tpr == my_tpr).all())

'''
    def test_equals_with_weights_roc(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            trues = trues > 0
            pred = pred[:,0]
            sk_fpr, sk_tpr, sk_threshold = roc_curve(trues, pred, sample_weight=weights)
            my_fpr, my_tpr, _  = get_roc_data(trues,pred,thresholds=sk_threshold, errors = False, weights=weights)
            self.assertTrue((sk_fpr == my_fpr).all())
            self.assertTrue((sk_tpr == my_tpr).all())
'''

if __name__ == '__main__':
    unittest.main()
