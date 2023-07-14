import sys
import unittest
import numpy as np
from metrices import *
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve


def terminal_decorator(func):
    def wrap(*args, **kwargs):
        print('#~'*20 +'\n')
        print(f"{func.__name__} is running")
        result = func(*args, **kwargs)
        print(f"{func.__name__} complete!")
        print('\n' + '#~'*20)
        return result
    return wrap

class Test_Conf_Matrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test = [np.random.randint(0,10*i, size=(1000*i,5)) for i in range(1, 10)]
        cls.trues = [np.random.randint(0,5,size= 1000*i) for i in range(1, 10)]
        cls.pred = [np.divide(test.T, np.sum(test, axis=1)).T for test in cls.test]
        cls.weights = [np.random.randint(0,5, size = 1000*i) for i in range(1, 10)]
    
    @terminal_decorator
    def test_unvalid_inputs(self):
        for ind, (trues, pred) in enumerate(zip(self.trues, self.pred)):
            with self.assertRaises(ValueError):
                get_conf_matrix(np.array([1,0,1,1,0]), pred)
            with self.assertRaises(ValueError):
                get_conf_matrix(trues, pred[:100])
            print(f'\t test_equals_no_weights: Iteration {ind} complete!')
    
    @terminal_decorator
    def test_unvalid_weights(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            with self.assertRaises(ValueError):
                get_conf_matrix(trues, pred, weights=weights[:100])
            with self.assertRaises(ValueError):
                get_conf_matrix(trues, pred, weights=np.stack((weights, weights), axis = 0))
            print(f'\t test_equals_no_weights: Iteration {ind} complete!')

    @terminal_decorator
    def test_equals_no_weights(self):
        for ind, (trues, pred) in enumerate(zip(self.trues, self.pred)):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1))
            my_matrix = get_conf_matrix(trues,pred)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())
            print(f'\t test_equals_no_weights: Iteration {ind} complete!')

    @terminal_decorator
    def test_equals_with_weights(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1),weights=weights)
            my_matrix = get_conf_matrix(trues,pred, weights=weights)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())
            print(f'\t test_equals_no_weights: Iteration {ind} complete!')

    
    @terminal_decorator
    def test_equals_with_errors(self):
        for ind, (trues, pred, weights) in enumerate(zip(self.trues, self.pred, self.weights)):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis = 1),weights=weights)
            my_matrix = get_conf_matrix(trues,pred, weights=weights, errors=True)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())
            print(f'\t test_equals_no_errors: Iteration {ind} complete!')
    
    #TODO Add Tests for ROC
    
if __name__ == '__main__':
    unittest.main()
