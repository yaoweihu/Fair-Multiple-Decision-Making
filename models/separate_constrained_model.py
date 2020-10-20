import cvxpy as cvx
from models.unconstrained_model import UnconstrainedModel


class SeparateConstrainedModel(UnconstrainedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = kwargs['tau']
        self.tau_ = 0.0

    def train(self, s, x, y, **kwargs):
        x_ = self.pre_processing(s, x, **kwargs)
        w = cvx.Variable(x_.shape[1])
        # map (0,1) to (-1, 1) for y
        y_h = cvx.multiply(2 * y.values - 1, x_.values @ w)
        obj = cvx.Minimize(cvx.sum(self.cvx_phi(y_h)) / x_.shape[0])

        s_pos_num = kwargs['s_pos_num']  # num of instances who have s=s^+
        s_neg_num = kwargs['s_neg_num']  # num of instances who have s=s^-

        weight = s.apply(lambda val: 1.0 / s_pos_num if val == kwargs['s_pos_val'] else 1.0 / s_neg_num)
        # map (0,1) to (-1, 1) for s
        s_z = cvx.multiply(2 * s.values - 1, x_.values * w)
        cons = [
            (cvx.sum(cvx.multiply(weight.values, self.cvx_phi(-s_z)))) - 1 <= self.tau_,
            #1 - (cvx.sum(cvx.multiply(weight.values, self.cvx_phi(s_z)))) >= - self.tau_
        ]

        prob = cvx.Problem(obj, cons)
        prob.solve(solver=cvx.ECOS, feastol=1e-8, abstol=1e-8, reltol=1e-8, max_iters=5000, verbose=False, warm_start=False)
        return w.value

    def fit1(self, s, x, y, **kwargs):
        self.tau_ = self.tau[0]
        self.w1 = self.train(s, x, y, **kwargs)

    def fit2(self, s, x, y, **kwargs):
        self.tau_ = self.tau[1]
        self.w2 = self.train(s, x, y, **kwargs)
