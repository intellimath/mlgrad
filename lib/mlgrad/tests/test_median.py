
import unittest
import numpy as np
import mlgrad.inventory as inv

class MedianCase(unittest.TestCase):

    def test_median_1(self):
        for i in range(40):
            a = np.random.random(100)
            self.assertTrue(np.median(a) == inv.median_1d(a))
    #
    def test_median_2(self):
        for i in range(40):
            a = np.random.random(101)
            self.assertTrue(np.median(a) == inv.median_1d(a))

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MedianCase))
    return suite
