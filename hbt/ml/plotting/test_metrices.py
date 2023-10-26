import unittest
import numpy as np
from metrices import get_conf_matrix, binary_roc_data, roc_curve_data
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve


class Test_Conf_Matrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test = [np.random.randint(0, 10 * i, size=(1000 * i, 5)) for i in range(1, 10)]
        cls.trues = [np.random.randint(0, 5, size=1000 * i) for i in range(1, 10)]
        cls.pred = [np.divide(test.T, np.sum(test, axis=1)).T for test in cls.test]
        cls.weights = [np.random.randint(0, 5, size=1000 * i) for i in range(1, 10)]

    def test_unvalid_inputs_conf(self):
        for trues, pred in zip(self.trues, self.pred):
            with self.assertRaises(AssertionError):
                get_conf_matrix(true_labels=np.array([1, 0, 1, 1, 0]), model_output=pred)
            with self.assertRaises(AssertionError):
                get_conf_matrix(true_labels=trues, model_output=pred[:100])

    def test_unvalid_weights_conf(self):
        for trues, pred, weights in zip(self.trues, self.pred, self.weights):
            with self.assertRaises(AssertionError):
                get_conf_matrix(true_labels=trues, model_output=pred, sample_weights=weights[:100])
            with self.assertRaises(AssertionError):
                get_conf_matrix(true_labels=trues, model_output=pred,
                                sample_weights=np.stack((weights, weights), axis=0))

    def test_equals_no_weights(self):
        for trues, pred in zip(self.trues, self.pred):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis=1))
            my_matrix = get_conf_matrix(true_labels=trues, model_output=pred)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())

    def test_equals_with_weights(self):
        for trues, pred, weights in zip(self.trues, self.pred, self.weights):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis=1), weights=weights)
            my_matrix = get_conf_matrix(true_labels=trues, model_output=pred, sample_weights=weights)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())

    def test_equals_with_errors(self):
        for trues, pred, weights in zip(self.trues, self.pred, self.weights):
            tf_matrix = confusion_matrix(trues, np.argmax(pred, axis=1), weights=weights)
            my_matrix = get_conf_matrix(true_labels=trues, model_output=pred, sample_weights=weights, errors=True)
            self.assertTrue((tf_matrix.numpy() == my_matrix).all())

    def test_unvalid_inputs_roc(self):
        for trues, pred in zip(self.trues, self.pred):
            trues = trues > 0
            pred = pred[:, 0]
            with self.assertRaises(AssertionError):
                binary_roc_data(true_labels=np.array([1, 0, 1, 1, 0]), model_output_positive=pred)
            with self.assertRaises(AssertionError):
                binary_roc_data(true_labels=trues, model_output_positive=pred[:100])

    def test_unvalid_weights_roc(self):
        for trues, pred, weights in zip(self.trues, self.pred, self.weights):
            trues = trues > 0
            pred = pred[:, 0]
            with self.assertRaises(AssertionError):
                binary_roc_data(true_labels=trues, model_output_positive=pred, sample_weights=weights[:100])
            with self.assertRaises(AssertionError):
                binary_roc_data(true_labels=trues, model_output_positive=pred,
                                sample_weights=np.stack((weights, weights), axis=0))

    def test_equals_without_weights_roc(self):
        for trues, pred in zip(self.trues, self.pred):
            trues = trues > 0
            pred = pred[:, 0]
            sk_fpr, sk_tpr, sk_threshold = roc_curve(trues, pred)
            my_fpr, my_tpr, _ = binary_roc_data(true_labels=trues, model_output_positive=pred,
                                                thresholds=sk_threshold, errors=False)
            self.assertTrue((sk_fpr == my_fpr).all())
            self.assertTrue((sk_tpr == my_tpr).all())

    def test_equals_without_weights_roc_new_impl(self):
        for trues, pred in zip(self.trues, self.pred):
            trues = trues > 0
            pred = pred[:, 0]
            sk_fpr, sk_tpr, sk_threshold = roc_curve(trues, pred)
            my_fpr, my_tpr, _ = roc_curve_data(evaluation_type="OvR", true_labels=trues,
                                               model_output=pred, thresholds=sk_threshold,
                                               errors=False)["0_vs_rest"].values()
            self.assertTrue((sk_fpr == my_fpr).all())
            self.assertTrue((sk_tpr == my_tpr).all())

    def test_equals_with_weights_roc(self):
        for trues, pred, weights in zip(self.trues, self.pred, self.weights):
            trues = trues > 0
            pred = pred[:, 0]
            sk_fpr, sk_tpr, sk_threshold = roc_curve(trues, pred, sample_weight=weights)
            my_fpr, my_tpr, _ = binary_roc_data(true_labels=trues, model_output_positive=pred,
                                                thresholds=sk_threshold, errors=False, sample_weights=weights)
            # sklearn arrays must be casted since it saves the result as float64 when using weights
            self.assertTrue((sk_fpr.astype(dtype=np.float32) == my_fpr).all())
            self.assertTrue((sk_tpr.astype(dtype=np.float32) == my_tpr).all())

    def test_output_size_mdim_roc_curve(self):
        for trues, pred, weights in zip(self.trues, self.pred, self.weights):
            OvO = roc_curve_data(evaluation_type="OvO", true_labels=trues, model_output=pred, errors=False)
            OvR = roc_curve_data(evaluation_type="OvR", true_labels=trues, model_output=pred, errors=False)
            self.assertEqual(20, len(OvO.keys()))
            self.assertEqual(5, len(OvR.keys()))


if __name__ == "__main__":
    unittest.main()
