# coding: utf-8

import unittest
from mlgrad.funcs import Square, Absolute
from mlgrad.models import ScaleLayer
from mlgrad.regr import regression
import numpy as np

# from matplotlib import pyplot as plt

class ScaleLayerCase(unittest.TestCase):

    def setUp(self):
        self.mod = ScaleLayer(Square(), 2)
        self.X = np.random.random(2)
        # print(self.mod.matrix.shape)
    #
    def test_scalelayer_2(self):
        self.mod.forward(self.X)
        Y = self.X * self.X / 2
        dy = np.max(np.abs(np.subtract(Y, self.mod.output)))
        self.assertTrue(dy < 1.0e-10)
    #


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ScaleLayerCase))
    return suite


