import os
import math
import pickle
import numpy as np
from sklearn.model_selection import ShuffleSplit

from models.prob_computation import *


def compute_fairness(s, y, phase):
    """ Compute total effects of s on y """
    p_y_pos_s1 = sum(y[s == 1] == 1) / len(y[s == 1])
    p_y_pos_s0 = sum(y[s == 0] == 1) / len(y[s == 0])
    te = p_y_pos_s1 - p_y_pos_s0

    return te


def exact_validation(data, splits, random_state=0):
    train_folds, test_folds = [data], [data]
    return train_folds, test_folds


def cross_validation(data, splits, random_state=0):
    """ Computes the cross validation datasets """
    rs = ShuffleSplit(n_splits=splits, test_size=0.2, random_state=random_state)
    train_folds, test_folds = [], []
    for train_idx, test_idx in rs.split(data):
        train_folds.append(data.iloc[train_idx])
        test_folds.append(data.iloc[test_idx])
    return train_folds, test_folds


def save_dict(data, file):
    """ Save prob_dict """
    if not os.path.exists(file):
        prob_dict = compute_probs(data)
        with open(file, 'wb') as f:
            pickle.dump(prob_dict, f)


def save_dict_adult(data, file):
    """ Save prob_dict """
    if not os.path.exists(file):
        prob_dict = compute_probs_adult(data)
        with open(file, 'wb') as f:
            pickle.dump(prob_dict, f)


def load_dict(file):
    """ Load prob_dict """
    if os.path.exists(file):
        with open(file, 'rb') as f:
            prob_dict = pickle.load(f)
        return prob_dict
    else:
        raise FileExistsError


def display_result(res, comment, logging):
    acc1 = np.array([val[0][0] for val in res])
    acc2 = np.array([val[1][0] for val in res])
    fair1 = np.array([val[0][1] for val in res])
    fair2 = np.array([val[1][1] for val in res])
    
    logging.info('-' * 20 + comment + '-' * 20)
    logging.info("Acc1: {:.2f}% +/- {:.2f}, Fair1: {:.2f} +/- {:.2f}".format(
        100 * np.mean(acc1), np.std(acc1), np.mean(fair1), np.std(fair1)))
    logging.info("Acc2: {:.2f}% +/- {:.2f}, Fair2: {:.2f} +/- {:.2f}\n".format(
        100 * np.mean(acc2), np.std(acc2), np.mean(fair2), np.std(fair2)))

    return [np.mean(acc1), np.mean(fair1)], [np.mean(acc2), np.mean(fair2)]


def save_train_test(train, test, file):

    df = pd.DataFrame({'train_acc1': [train[0][0]], 'train_fair1': [train[0][1]],
                       'train_acc2': [train[1][0]], 'train_fair2': [train[1][1]],
                       'test_acc1': [test[0][0]], 'test_fair1': [test[0][1]],
                       'test_acc2': [test[1][0]], 'test_fair2': [test[1][1]]})
    if not os.path.exists(file):
        df.to_csv(file, mode='w', header=True, index=False)
    else:
        df.to_csv(file, mode='a', header=False, index=False)