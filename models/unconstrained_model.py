from models.base_model import *


class UnconstrainedModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w1 = 0
        self.w2 = 0

    def pre_processing(self, s, x, **kwargs):
        # encode x, add intercept, combine x and s
        # necessary before fit and predict
        if isinstance(x, pd.DataFrame) and set(kwargs['dummy_cols']).issubset(set(x.columns)):
            x_ = pd.get_dummies(x, columns=kwargs['dummy_cols'])  # force selected columns to be converted
        else:
            x_ = pd.get_dummies(x)
        x_ = self.add_intercept(x_)
        if 'enable_s' in kwargs.keys() and kwargs['enable_s']:
            # if enable_s is not given, s is not used.
            x_ = pd.concat([s, x_], axis=1)
        return x_

    def train(self, s, x, y, **kwargs):
        x_ = self.pre_processing(s, x, **kwargs)
        w = cvx.Variable(x_.shape[1])
        # map (0,1) to (-1, 1) for y
        y_h = cvx.multiply(2 * y.values - 1, x_.values @ w)
        obj = cvx.Minimize(cvx.sum(self.cvx_phi(y_h)) / x_.shape[0])

        prob = cvx.Problem(obj)
        prob.solve(solver=cvx.ECOS, feastol=1e-8, abstol=1e-8, reltol=1e-8, max_iters=5000, verbose=False, warm_start=False)
        return w.value

    def fit1(self, s, x, y, **kwargs):
        self.w1 = self.train(s, x, y, **kwargs)

    def fit2(self, s, x, y, **kwargs):
        self.w2 = self.train(s, x, y, **kwargs)

    def predict1(self, s, x, **kwargs):
        x_ = self.pre_processing(s, x, **kwargs)
        pred = np.int32((x_.values @ self.w1) >= 0)
        return pd.Series(pred, index=x.index)  # return a pandas.Series

    def predict2(self, s, x, **kwargs):
        x_ = self.pre_processing(s, x, **kwargs)
        pred = np.int32((x_.values @ self.w2) >= 0)
        return pd.Series(pred, index=x.index)  # return a pandas.Series
