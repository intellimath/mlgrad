#
# irls.py
#

class IR_ERM:
    
    def fit(self):
        
        self.risk.evaluate_losses(self.lval_all)
        self.gd.fit()