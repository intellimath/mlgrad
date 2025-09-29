#
# irls.py
#

class IR_ERM:

    def fit(self):

        lval_all = self.risk.evaluate_losses()
        lval = self.aggfunc.evaluate(lval_all)
        self.gd.fit()