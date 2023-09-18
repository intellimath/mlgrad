# coding: utf-8

import unittest
from mlgrad.func import Square, Absolute
from mlgrad.model import LinearLayer
from mlgrad.regr import regression
import numpy as np

# from matplotlib import pyplot as plt

class LinearLayerCase(unittest.TestCase):

    def setUp(self):
        self.mod = LinearLayer(2, 5)
        self.mod.allocate()
        self.mod.init_param()
        self.X = np.random.random(2)
        # print(self.mod.matrix.shape)

    def test_linearlayer_1(self):
        self.assertEqual(self.mod.n_output, 5)
        self.assertEqual(self.mod.n_input, 2)
        self.assertEqual(self.mod.n_param, 15)
    #
    def test_linearlayer_2(self):
        self.mod.forward(self.X)
        Y = np.asarray(self.mod.matrix[:,0]) + np.dot(self.mod.matrix[:,1:], self.X)
        dy = np.max(np.abs(np.subtract(Y, self.mod.output)))
        self.assertTrue(dy < 1.0e-10)
    #
    def test_linearlayer_3(self):
        grad_out = 2 * np.ones(self.mod.n_output, 'd')
        grad = np.zeros(self.mod.n_param, 'd')
        self.mod.backward(self.X, grad_out, grad)
        print()
        print(np.asarray(2*self.X), np.asarray(grad))
        print(np.dot(grad_out, self.mod.matrix)[1:], np.asarray(self.mod.grad_input))
    #


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LinearLayerCase))
    return suite


