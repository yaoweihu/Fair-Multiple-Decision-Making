from models.unconstrained_model import *
from models.separate_constrained_model import *
from models.serial_constrained_model import *
from models.joint_constrained_model import *
from synthetic.synthetic_generator import *



def run_unconstrained_model(**kwargs):
    clf = UnconstrainedModel(**kwargs)

    s_name = kwargs['s_name']
    x1_name = kwargs['x1_name']
    y1_name = kwargs['y1_name']
    x2_name = kwargs['x2_name']
    y2_name = kwargs['y2_name']

    # -------------- train 1 --------------
    s_train = kwargs['train_data'][s_name]
    x1_train = kwargs['train_data'][x1_name]
    y1_train = kwargs['train_data'][y1_name]

    clf.fit1(s_train, x1_train, y1_train)
    y1_train_pred = clf.predict1(s_train, x1_train)
    res_y1_train_pred = clf.eval(y1_train, y1_train_pred, s_train)

    # -------------- train 2 ---------------
    x2_train = kwargs['train_data'][x2_name]
    x12_train = pd.concat([x1_train, y1_train, x2_train], axis=1)
    y2_train = kwargs['train_data'][y2_name]

    clf.fit2(s_train, x12_train, y2_train, dummy_cols=['x2'])
    y2_train_pred = clf.predict2(s_train, x12_train, dummy_cols=['x2'])
    res_y2_train_pred = clf.eval(y2_train, y2_train_pred, s_train)

    # -------------- test 1 ---------------
    s_test = kwargs['test_data'][s_name]
    x1_test = kwargs['test_data'][x1_name]
    y1_test = kwargs['test_data'][y1_name]

    y1_test_pred = clf.predict1(s_test, x1_test)
    res_y1_test_pred = clf.eval(y1_test, y1_test_pred, s_test)

    # -------------- test 2 ---------------
    test_data_new = generate_eval_data(s_test, x1_test, y1_test_pred)
    x2_test = test_data_new[x2_name]
    x12_test = pd.concat([x1_test, y1_test_pred, x2_test], axis=1)
    y2_test = test_data_new[y2_name]

    y2_test_pred = clf.predict2(s_test, x12_test, dummy_cols=['x2'])
    res_y2_test_pred = clf.eval(y2_test, y2_test_pred, s_test)

    return [res_y1_train_pred, res_y2_train_pred], [res_y1_test_pred, res_y2_test_pred]


def run_separate_model(**kwargs):
    clf = SeparateConstrainedModel(**kwargs)

    s_name = kwargs['s_name']
    x1_name = kwargs['x1_name']
    y1_name = kwargs['y1_name']
    x2_name = kwargs['x2_name']
    y2_name = kwargs['y2_name']

    # -------------- train 1 --------------
    s_train = kwargs['train_data'][s_name]
    s_pos_num = s_train.sum()
    s_neg_num = s_train.shape[0] - s_pos_num
    x1_train = kwargs['train_data'][x1_name]
    y1_train = kwargs['train_data'][y1_name]

    clf.fit1(s_train, x1_train, y1_train, s_pos_num=s_pos_num, s_neg_num=s_neg_num, s_pos_val=1)
    y1_train_pred = clf.predict1(s_train, x1_train)
    res_y1_train_pred = clf.eval(y1_train, y1_train_pred, s_train)

    # -------------- train 2 ---------------
    x2_train = kwargs['train_data'][x2_name]
    x12_train = pd.concat([x1_train, y1_train, x2_train], axis=1)
    y2_train = kwargs['train_data'][y2_name]

    clf.fit2(s_train, x12_train, y2_train, s_pos_num=s_pos_num, s_neg_num=s_neg_num, s_pos_val=1, dummy_cols=['x2'])
    y2_train_pred = clf.predict2(s_train, x12_train, dummy_cols=['x2'])
    res_y2_train_pred = clf.eval(y2_train, y2_train_pred, s_train)

    # -------------- test 1 ---------------
    s_test = kwargs['test_data'][s_name]
    x1_test = kwargs['test_data'][x1_name]
    y1_test = kwargs['test_data'][y1_name]

    y1_test_pred = clf.predict1(s_test, x1_test)
    res_y1_test_pred = clf.eval(y1_test, y1_test_pred, s_test)

    # -------------- test 2 ---------------
    test_data_new = generate_eval_data(s_test, x1_test, y1_test_pred)
    x2_test = test_data_new[x2_name]
    x12_test = pd.concat([x1_test, y1_test_pred, x2_test], axis=1)
    y2_test = test_data_new[y2_name]

    y2_test_pred = clf.predict2(s_test, x12_test, dummy_cols=['x2'])
    res_y2_test_pred = clf.eval(y2_test, y2_test_pred, s_test)

    return [res_y1_train_pred, res_y2_train_pred], [res_y1_test_pred, res_y2_test_pred]


def run_serial_model(**kwargs):
    clf = SerialConstrainedModel(**kwargs)

    s_name = kwargs['s_name']
    x1_name = kwargs['x1_name']
    y1_name = kwargs['y1_name']
    x2_name = kwargs['x2_name']
    y2_name = kwargs['y2_name']

    # -------------- train 1 --------------
    s_train = kwargs['train_data'][s_name]
    s_pos_num = s_train.sum()
    s_neg_num = s_train.shape[0] - s_pos_num
    x1_train = kwargs['train_data'][x1_name]
    y1_train = kwargs['train_data'][y1_name]

    clf.fit1(s_train, x1_train, y1_train, s_pos_num=s_pos_num, s_neg_num=s_neg_num, s_pos_val=1)
    y1_train_pred = clf.predict1(s_train, x1_train)
    res_y1_train_pred = clf.eval(y1_train, y1_train_pred, s_train)

    # -------------- train 2 ---------------
    train_data_new = generate_eval_data(s_train, x1_train, y1_train_pred)
    x2_train = train_data_new[x2_name]
    y1_train_ = train_data_new[y1_name]
    x12_train = pd.concat([x1_train, y1_train_, x2_train], axis=1)
    y2_train = train_data_new[y2_name]

    clf.fit2(s_train, x12_train, y2_train, s_pos_num=s_pos_num, s_neg_num=s_neg_num, s_pos_val=1, dummy_cols=['x2'])
    y2_train_pred = clf.predict2(s_train, x12_train, dummy_cols=['x2'])
    res_y2_train_pred = clf.eval(y2_train, y2_train_pred, s_train)

    # -------------- test 1 ---------------
    s_test = kwargs['test_data'][s_name]
    x1_test = kwargs['test_data'][x1_name]
    y1_test = kwargs['test_data'][y1_name]

    y1_test_pred = clf.predict1(s_test, x1_test)
    res_y1_test_pred = clf.eval(y1_test, y1_test_pred, s_test)

    # -------------- test 2 ---------------
    test_data_new = generate_eval_data(s_test, x1_test, y1_test_pred)
    x2_test = test_data_new[x2_name]
    x12_test = pd.concat([x1_test, y1_test_pred, x2_test], axis=1)
    y2_test = test_data_new[y2_name]

    y2_test_pred = clf.predict2(s_test, x12_test, dummy_cols=['x2'])
    res_y2_test_pred = clf.eval(y2_test, y2_test_pred, s_test)

    return [res_y1_train_pred, res_y2_train_pred], [res_y1_test_pred, res_y2_test_pred]


def run_joint_model(**kwargs):
    clf = JointConstrainedModel(**kwargs)

    s_name = kwargs['s_name']
    x1_name = kwargs['x1_name']
    y1_name = kwargs['y1_name']
    x2_name = kwargs['x2_name']
    y2_name = kwargs['y2_name']

    # -------------- train 1 --------------
    s_train = kwargs['train_data'][s_name]
    x1_train = kwargs['train_data'][x1_name]
    y1_train = kwargs['train_data'][y1_name]
    x2_train = kwargs['train_data'][x2_name]
    y2_train = kwargs['train_data'][y2_name]
    x12_train = pd.concat([y1_train, x2_train], axis=1)

    res_train = clf.fit(s_train, x1_train, x12_train)
    
    # -------------- test 1 ---------------
    s_test = kwargs['test_data'][s_name]
    x1_test = kwargs['test_data'][x1_name]
    y1_test = kwargs['test_data'][y1_name]

    y1_test_pred = clf.predict1(s_test, x1_test, dummy_cols=['x1'])
    res_y1_test_pred = clf.eval(y1_test, y1_test_pred, s_test)

    # -------------- test 2 ---------------
    test_data_new = generate_eval_data(s_test, x1_test, y1_test_pred)
    x2_test = test_data_new[x2_name]
    x12_test = pd.concat([y1_test_pred, x2_test], axis=1)
    y2_test = test_data_new[y2_name]

    y2_test_pred = clf.predict2(s_test, x12_test, dummy_cols=['x2'])
    res_y2_test_pred = clf.eval(y2_test, y2_test_pred, s_test)

    return res_train, [res_y1_test_pred, res_y2_test_pred]