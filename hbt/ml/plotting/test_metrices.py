import unittest
import numpy as np
from metrices import *
from tensorflow.math import confusion_matrix

class Test_Conf_Matrix(unittest.TestCase):
    
    def test_equals_no_weights(self):
        test = np.random.randint(0,100, size=(1000,5))
        trues = np.random.randint(0,5,size= 1000)
        pred = np.divide(test.T, np.sum(test, axis=1)).T
        weights = np.random.randint(0,5, size = 1000)
        tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1))
        my_matrix = get_conf_matrix(trues,pred)
        self.assertTrue((tf_matrix.numpy() == my_matrix).all())

    def test_equals_with_weights(self):
        test = np.random.randint(0,100, size=(1000,5))
        trues = np.random.randint(0,5,size= 1000)
        pred = np.divide(test.T, np.sum(test, axis=1)).T
        weights = np.random.randint(0,5, size = 1000)
        tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1),weights=weights)
        my_matrix = get_conf_matrix(trues,pred, weights=weights)
        self.assertTrue((tf_matrix.numpy() == my_matrix).all())

if __name__ == '__main__':
    unittest.main()