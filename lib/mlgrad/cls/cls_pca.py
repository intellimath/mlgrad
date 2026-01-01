import numpy as np
import mlgrad.regr as regr
import mlgrad.models as models
import mlgrad.inventory as inventory

import matplotlib.pyplot as plt

def normalize(a):
    return a / np.sqrt(a @ a)

def make_model(n):
    mod = models.LinearModel(n)
    return mod

def accuracy_score(Y1, Y2):
    return (Y1 == Y2).astype("d").mean()

def cls_pca(X, Y, lossfunc, m=2, model_maker=make_model,
             regnorm=None, tau=0.0,
             normalizer=None, support_negate=True,
             verbose=True, callback=None, h=0.001, n_iter=1000, tol=1.0e-9):
    As = []
    Us = []
    mods = []
    XX, YY = X, Y
    N, n = XX.shape
    cc = 0

    for k in range(m):
        mod = model_maker(n)
        if regnorm is not None and tau != 0:
            mod.use_regularizer(regnorm, tau)
        alg = regr.regression(XX, YY, mod, lossfunc,
                  normalizer=normalizer,
                  # regnorm=regnorm, tau=tau,
                  h=h, n_iter=n_iter, tol=tol)

        inventory.normalize2(mod.param)

        if support_negate:
            U = mod.evaluate(XX)
            score = accuracy_score(np.sign(U), YY)
            if score < 0.5:
                for i,v in enumerate(mod.param):
                    mod.param[i] = -v

        a = np.array(mod.param, copy=True)
        a1 = a[1:]
        for aa in As:
            aa1 = aa[1:]
            a1 -= (aa1 @ a1) * aa1
        norm_a = np.sqrt(a1 @ a1)
        if norm_a == 0:
            break
        a /= norm_a
        for i in range(len(mod.param)):
            mod.param[i] = a[i]
        As.append(a)


        U = mod.evaluate(XX)
        if verbose:
            print(k, ":", alg.K, accuracy_score(np.sign(U), YY))
        Us.append(U)
        mods.append(mod)


        if callback is not None:
            callback(XX, YY, mod)

        a1 = a[1:]
        c = a[0]
        ca = c * a1
        cc += ca
        XX = XX - ca
        XX = XX - np.outer(XX @ a1, a1)
        # XX = np.array([(xx - (xx @ a1) * a1) for xx in XX])

    As = np.array(As)
    Us = np.array(Us)
    return cc, As, Us, mods