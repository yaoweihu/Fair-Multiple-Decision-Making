import os.path
import numpy as np
from itertools import product
from random import random

from models.utils import *



np.random.seed(0)
s_name = 'age'
x1_name = ['sex', 'education', 'marital-status']
y1_name = 'workclass'
x2_name = 'hours'
y2_name = 'income'


def generate_phase1(data):
    x1 = pd.Series(np.dot(data[x1_name], np.array([4, 2, 1])))
    return data[s_name], x1, data[y1_name]


def generate_phase2(data, s, x1, y1):
    x2 = data[x2_name]
    y2 = data[y2_name]
    return x2, y2


def generate_data(n_samples=5000):
    np.random.seed(0)
    data = pd.read_csv('../data/adult_binary.csv')
    s, x1, y1 = generate_phase1(data)
    x2, y2 = generate_phase2(data, s, x1, y1)

    return pd.DataFrame({'s': s, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})


def generate_eval_data(s, x1, y1, **kwargs):
    # lear to predict w2 and y2
    cpt = ProbTable(kwargs['distribution'])
    r = kwargs['random_state']

    def make_decision(s, x1, y1):
        x2_p1 = cpt.get_probabilty(variables={'x2': 1}, conditions={'s': s, 'x1': x1, 'y1': y1})
        x2 = 1 if r.rand(1)[0] < x2_p1 else 0

        y2_p1 = cpt.get_probabilty(variables={'y2': 1}, conditions={'s': s, 'x1': x1, 'y1': y1, 'x2': x2})
        y2 = 1 if r.rand(1)[0] < y2_p1 else 0
        return [x2, y2]

    df = pd.DataFrame({'s': s, 'x1': x1, 'y1': y1})
    x2_y2 = df.apply(lambda l: make_decision(l['s'], l['x1'], l['y1']), axis=1, result_type='expand')
    return pd.DataFrame({'s': s, 'x1': x1, 'x2': x2_y2[0], 'y1': y1, 'y2': x2_y2[1]})


def cal_statistic(data):
    print("s:", dict(data['s'].value_counts()))
    print("x1 when s=0:", dict(data[data['s'] == 0]['x1'].value_counts()))
    print("x1 when s=1:", dict(data[data['s'] == 1]['x1'].value_counts()))
    print("y1 when s=0:", dict(data[data['s'] == 0]['y1'].value_counts()))
    print("y1 when s=1:", dict(data[data['s'] == 1]['y1'].value_counts()))
    print("x2 when s=0:", dict(data[data['s'] == 0]['x2'].value_counts()))
    print("x2 when s=1:", dict(data[data['s'] == 1]['x2'].value_counts()))
    print("y2 when s=0:", dict(data[data['s'] == 0]['y2'].value_counts()))
    print("y2 when s=1:", dict(data[data['s'] == 1]['y2'].value_counts()))


if __name__ == "__main__":
    data = generate_data(n_samples=0)
    cal_statistic(data)