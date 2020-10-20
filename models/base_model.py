import numpy as np
import pandas as pd
import cvxpy as cvx


class BaseModel:

    def __init__(self, **kwargs):
        self.logging = kwargs['logging']
        self.choose_phi(kwargs['phi_name'])

    def choose_phi(self, phi_name):
        if phi_name == 'logistic':
            self.cvx_phi = lambda z: cvx.logistic(-z)
        elif phi_name == 'hinge':
            self.cvx_phi = lambda z: cvx.pos(1.0 - z)
        elif phi_name == 'exponential':
            self.cvx_phi = lambda z: cvx.exp(-z)
        else:
            print("Your surrogate function doesn't exist.")

    def add_intercept(self, x):
        inter_ = pd.Series(1, x.index, name='intercept')
        return pd.concat([x, inter_], axis=1)

    def eval(self, y_truth, y_pred, s):
        acc = sum(y_truth == y_pred) / len(y_truth)

        df = pd.DataFrame({'s': s, 'truth': y_truth, 'pred': y_pred})
        p1 = len(df.query('pred == 1 & s == 1')) / len(df.query('s == 1'))
        p0 = len(df.query('pred == 1 & s == 0')) / len(df.query('s == 0'))
        fair = p1 - p0

        return acc, fair
