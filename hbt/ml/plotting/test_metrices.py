import unittest
import numpy as np
from metrices import *
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve


class Test_Conf_Matrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test = np.random.randint(0,100, size=(1000,5))
        cls.trues = np.random.randint(0,5,size= 1000)
        cls.pred = np.divide(test.T, np.sum(test, axis=1)).T
        cls.weights = np.random.randint(0,5, size = 1000)

    def test_unvalid_inputs(self):
        true_labels = np.array([1,0,1,0])
        output = np.random
        with self.assertRaises(ValueError):
            get_conf_matrix(np.array([1,0,1,1,0]), self.pred)
        with self.assertRaises(ValueError):
            get_conf_matrix(self.trues, self.pred[:100])

    def test_unvalid_weights(self):
        pass

    def test_equals_no_weights(self):
        tf_matrix = confusion_matrix(self.trues, np.argmax(self.pred, axis = 1))
        my_matrix = get_conf_matrix(self.trues,self.pred)
        self.assertTrue((tf_matrix.numpy() == my_matrix).all())

    def test_equals_with_weights(self):
        tf_matrix = confusion_matrix(self.trues, np.argmax(self.pred, axis = 1),weights=self.weights)
        my_matrix = get_conf_matrix(self.trues,self.pred, weights=self.weights)
        self.assertTrue((tf_matrix.numpy() == my_matrix).all())

if __name__ == '__main__':
    unittest.main()
