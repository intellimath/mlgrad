import numpy as np
import mlgrad.regr as regr
import mlgrad.models as models
import mlgrad.funcs2 as funcs2
import mlgrad.inventory as inventory

import matplotlib.pyplot as plt

def normalize(a):
    aa = np.asarray(a)
    aa1 = aa[1:]
    a[:] = aa / np.sqrt(aa1 @ aa1)

def make_model(n):
    mod = models.LinearModel(n)
    return mod

def accuracy_score(Y1, Y2):
    return (Y1 == Y2).astype("d").mean()

def cls_pca(X, Y, lossfunc, m=2, model_maker=make_model,
             regnorm=None, tau=0,
             normalizer=None, support_negate=False, othogonalize=True,
             verbose=True, callback=None, h=0.001, n_iter=1000, tol=1.0e-9):
    As = []
    Us = []
    mods = []
    XX, YY = X, Y
    N, n = XX.shape
    cc = 0

    zz = np.zeros(n, "d")

    for k in range(m):
        mod = model_maker(n)
        mod.init_param()
        if regnorm is not None and tau != 0:
            mod.use_regularizer(regnorm, tau)
        # if len(As) > 0:
        #     for aa in As:
        #         eqn = funcs2.Dot(aa, 1)
        #         mod.use_eqn(eqn, 0)
        alg = regr.regression(XX, YY, mod, lossfunc,
                  normalizer=normalizer,
                  h=h, n_iter=n_iter, tol=tol)

        # normalize(mod.param)

        if support_negate:
            U = mod.evaluate(XX)
            score = accuracy_score(np.sign(U), YY)
            if score < 0.5:
                for i,v in enumerate(mod.param):
                    mod.param[i] = -v

        # U = mod.evaluate(XX)
        # if verbose:
        #     print(k, ":", alg.K, accuracy_score(np.sign(U), YY))

        a = np.array(mod.param, copy=True)
        # normalize parameters
        a1 = a[1:]
        a /= np.sqrt(a1 @ a1)

        # find center in the hyperplane
        c = a[0]
        ca = c * a1

        if othogonalize:
            # orthogonalize a1 against vectors in As
            for aa in As:
                aa1 = aa[1:]
                a1 -= (aa1 @ a1) * aa1

            # renormalize a1
            a1 /= np.sqrt(a1 @ a1)

        mod.param[:] = a

        U = mod.evaluate(XX)
        if verbose:
            print(k, ":", alg.K, accuracy_score(np.sign(U), YY))

        if (abs(a1) >= 1.0e-10).astype("i").sum() == 0:
            print("Zeros array")
            break

        As.append(a)
        Us.append(U)
        mods.append(mod)

        if callback is not None:
            callback(XX, YY, mod, alg)

        ca = -c * a1
        cc += ca
        XX = XX - ca
        XX = XX - np.outer(XX @ a1, a1)

    As = np.asarray(As)
    # As = As[:,1:]
    Us = np.asarray(Us)
    return cc, As, Us, mods
