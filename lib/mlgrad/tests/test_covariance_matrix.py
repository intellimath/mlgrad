
import unittest
import numpy as np
import mlgrad.inventory as inv

class CovmatrCase(unittest.TestCase):

    def test_covmatr_1(self):
        for i in range(1):
            X = np.random.random(size=(100,5))
            c = np.random.random(size=5)
            Xc = X - c
            S1 = Xc.T @ Xc / 100
            S2 = inv.covariance_matrix(X, c)
            print(S1)
            print(S2.base)
            self.assertTrue(np.allclose(S1, S2.base))
    #
    # def test_covmatr_2(self):
    #     for i in range(10):
    #         a = np.random.random(101)
    #         self.assertTrue(np.median(a) == inv.median_1d(a))

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CovmatrCase))
    return suite
