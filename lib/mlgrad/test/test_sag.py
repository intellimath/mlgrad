# coding: utf-8

import unittest
from mlgrad.sag import SAG
from mlgrad.func import Square, Absolute
from mlgrad.model import LinearModel
import numpy as np


class SAGCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_LeastMeanSquares(self):
        X = np.random.random(size=(100,4))
        coeff = np.array([1., 2., 3., 4., 5.])
        #
        model_orig = LinearModel(coeff)
        model = LinearModel(np.random.random(size=5))
        #
        Y = model_orig.evaluate_all(X)
        #
        loss_func = Square()
        sag = SAG(model, loss_func, tol=1.0e-7, h=1.0)
        sag.fit(X, Y)
        diff_coeff = np.abs(model_orig.param.base - model.param.base)
        print(sag.K, diff_coeff)
        self.assertTrue(all(diff_coeff < 1.0e-4))
    #
    def test_LeastMeanAbsolutes(self):
        X = np.random.random(size=(100,4))
        coeff = np.array([1., 2., 3., 4., 5.])
        #
        model_orig = LinearModel(coeff)
        model = LinearModel(np.random.random(size=5))
        #
        Y = model_orig.evaluate_all(X)
        #
        loss_func = Absolute()
        sag = SAG(model, loss_func, tol=1.0e-6, h=0.001)
        sag.fit(X, Y)
        diff_coeff = np.abs(model_orig.param.base - model.param.base)
        print(sag.K, diff_coeff)
        self.assertTrue(all(diff_coeff < 1.0e-4))
    #


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SAGCase))
    return suite


