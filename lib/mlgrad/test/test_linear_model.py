# coding: utf-8

import unittest
from mlgrad.funcs import Square, Absolute
from mlgrad.models import LinearModel
from mlgrad.regr import regression
import numpy as np

# from matplotlib import pyplot as plt

class LinearModelCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_1(self):
        mod = LinearModel(2)
        mod.init_param(random=False)
        self.assertEqual(mod.n_param, 3)
        self.assertEqual(mod.n_input, 2)
        self.assertEqual(mod.param[0], 0)
        self.assertEqual(mod.param[1], 0)
        self.assertEqual(mod.param[2], 0)
    #
    def test_2(self):
        mod = LinearModel(2)
        mod.init_param(np.array([1,2,3], 'd'))
        self.assertEqual(mod.n_param, 3)
        self.assertEqual(mod.n_input, 2)
        self.assertEqual(mod.param[0], 1.)
        self.assertEqual(mod.param[1], 2.)
        self.assertEqual(mod.param[2], 3.)
    #
    def test_3(self):
        mod = LinearModel(np.array([1,2,3], 'd'))
        self.assertEqual(mod.n_param, 3)
        self.assertEqual(mod.n_input, 2)
        self.assertEqual(mod.param[0], 1.)
        self.assertEqual(mod.param[1], 2.)
        self.assertEqual(mod.param[2], 3.)
    #
    def test_4(self):
        mod = LinearModel(np.array([1,2,3], 'd'))
        self.assertEqual(mod.evaluate(np.array([1,1], 'd')), 6.)
        self.assertEqual(mod.evaluate(np.array([2,3], 'd')), 14.)
    #
    def test_5(self):
        mod = LinearModel(np.array([1,2,3], 'd'))
        X = np.array([1,2], 'd')
        self.assertTrue(any(mod.gradient(X) == np.array([1,1,2], 'd')))
    #
    def test_6(self):
        mod = LinearModel(np.array([1,2,3], 'd'))
        X = np.array([1,2], 'd')
        self.assertTrue(any(mod.gradient_input(X) == np.array([2,3], 'd')))
    #
    def test_ols_1(self):
        X = np.array([[-1.],[-0.5], [0], [0.5], [1.]], 'd')
        Y = np.array([-1., 0, 1., 2., 3.], 'd')
        mod = LinearModel(1)
        mod.init_param()
        alg = regression(X, Y, mod, h=0.001, n_iter=5000, tol=1.0e-10, n_restart=5)
        print(alg.K, np.array([1.,2.]), np.asarray(mod.param))
        # plt.loglog(alg.lvals)
        # plt.show()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LinearModelCase))
    return suite


