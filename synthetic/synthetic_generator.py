import numpy as np
import pandas as pd
np.random.seed(0)


def make_decision_y1(ts, tx1):
    r = np.random.rand()
    if tx1 + 3 * ts > 7:
        return 1 if r < 0.9 else 0
    else:
        return 0 if r < 0.9 else 1


def make_decision_y2(ty1, tx2):
    r = np.random.rand()
    if tx2 + 3 * ty1 > 6:
        return 1 if r < 0.9 else 0
    else:
        return 0 if r < 0.9 else 1


def make_decision_x2(ts, ty1, tx1):
    if ts == 1:
        p0 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 10]).reshape(10, 10)
        p1 = np.array([[1, 1, 1, 1, 3, 3, 3, 3, 3, 3] * 10]).reshape(10, 10)
    if ts == 0:
        p0 = np.array([[1, 3, 3, 1, 1, 1, 1, 1, 1, 1] * 10]).reshape(10, 10)
        p1 = np.array([[1, 1, 1, 1, 3, 3, 3, 3, 3, 3] * 10]).reshape(10, 10)
    
    p0 = p0 / p0.sum(axis=1, keepdims=True)
    p1 = p1 / p1.sum(axis=1, keepdims=True)
    if ty1 == 0:
        res = np.random.multinomial(1, p0[int(tx1)])
    if ty1 == 1:
        res = np.random.multinomial(1, p1[int(tx1)])
    return np.argwhere(res == 1)[0][0]


def generate_phase1(n_samples=5000):
    s0 = np.zeros(n_samples, dtype=np.int32)  
    s1 = np.ones(n_samples, dtype=np.int32)  

    p1 = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    p0 = [1, 1, 1, 1, 2, 1, 2, 1, 1, 2]

    p0 = np.array(p0)
    p1 = np.array(p1)
    p0 = p0 / p0.sum()
    p1 = p1 / p1.sum()

    x0 = np.random.choice(10, size=n_samples, p=p0).astype(np.int32)
    x1 = np.random.choice(10, size=n_samples, p=p1).astype(np.int32)

    s = np.hstack((s0, s1))
    x = np.hstack((x0, x1))
    y = np.array([make_decision_y1(ts, tx) for ts, tx in zip(s, x)], dtype=np.int32)

    perm = list(range(0, n_samples * 2))
    np.random.shuffle(perm)

    return s[perm], x[perm], y[perm]


def generate_phase2(s, x1, y1, n_samples=5000):
    x2 = np.array([make_decision_x2(ts, ty1, tx1) for ts, ty1, tx1 in zip(s, y1, x1)], dtype=np.int32)
    y2 = np.array([make_decision_y2(ty1, tx2) for ty1, tx2 in zip(y1, x2)], dtype=np.int32)

    return x2, y2


def generate_synthetic_data(n_samples=5000):
    s, x1, y1 = generate_phase1(n_samples)
    x2, y2 = generate_phase2(s, x1, y1, n_samples * 2)

    return pd.DataFrame({'s': s, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})


def generate_eval_data(s, x1, y1):
    np.random.seed(0)
    x2, y2 = generate_phase2(s, x1, y1, s.shape[0])
    return pd.DataFrame({'s': s, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})


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
    data = generate_synthetic_data()
    cal_statistic(data)