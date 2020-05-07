# coding: utf-8

import unittest
from mlgrad.avragg import Average_GD
from mlgrad.func import Quantile, Expectile
import numpy as np


class Average_GDCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_Quantile_50(self):
        X = np.random.random(size=101)
        #
        dsm_func = Quantile(0.5)
        sagagg = Average_GD(dsm_func, tol=1.0e-5, h=0.01)
        sagagg.fit(X)
        diff = np.abs(sagagg.y - np.median(X))
        print(sagagg.K, diff)
        #self.assertTrue(diff < 1.0e-3)
    #
    def test_Expectile_50(self):
        X = np.random.random(size=101)
        #
        dsm_func = Expectile(0.5)
        sagagg = Average_GD(dsm_func, tol=1.0e-6, h=0.1)
        sagagg.fit(X)
        diff = np.abs(sagagg.y - np.mean(X))
        print(sagagg.K, diff)
        #self.assertTrue(diff < 1.0e-3)
    #


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Average_GDCase))
    return suite


