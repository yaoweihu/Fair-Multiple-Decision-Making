import sys
import time
import logging
import argparse
sys.path.append('..')

from models import utils
from run_synthetic_models import *



if __name__ == '__main__':

    SEED = 0
    NUM_FOLDS = 5
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="specify the model")
    args = parser.parse_args()

        
    r = np.random.seed(SEED)
    t_start = time.time()
    # ---------- output file ------------------------------------
    file = '../results/synthetic_' + args.model + '.log'
    logging.basicConfig(filename=file, level=logging.INFO)
    logging.info('random seed: %d' % SEED)
    logging.info(args.model)

    # ---------- generate data ----------------------------------
    data = generate_synthetic_data(n_samples=5000)
    
    # ---------- compute original fairness ----------------------
    fair1 = utils.compute_fairness(data['s'], data['y1'], 0)
    fair2 = utils.compute_fairness(data['s'], data['y2'], 1)
    logging.info(f"Fairness on original dataset:\n"
                 f"Fairness of s on y1: {fair1:.2f}\n"
                 f"Fairness of s on y2: {fair2:.2f}\n")

    # ---------- cross validation ------------------------------
    train_folds, test_folds = utils.cross_validation(data, NUM_FOLDS, SEED)

    # ---------- training --------------------------------------
    tr_res, te_res = [], []

    for i, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds)):

        if args.model == 'unconstrained_model':
            tr, te = run_unconstrained_model(logging=logging, train_data=train_fold, test_data=test_fold,
                                             phi_name='logistic', enerate_eval_data=generate_eval_data,
                                             s_name='s', x1_name='x1', y1_name='y1', x2_name='x2', y2_name='y2')
        if args.model == 'separate_constrained_model':
            tr, te = run_separate_model(logging=logging, train_data=train_fold, test_data=test_fold,
                                        generate_eval_data=generate_eval_data, s_name='s', x1_name='x1', 
                                        y1_name='y1', x2_name='x2', y2_name='y2',
                                        phi_name='logistic', tau=[0.45, 0.46])
        if args.model == 'serial_constrained_model':
            tr, te = run_serial_model(logging=logging, train_data=train_fold, test_data=test_fold,
                                      generate_eval_data=generate_eval_data, s_name='s', x1_name='x1', 
                                      y1_name='y1', x2_name='x2', y2_name='y2', 
                                      phi_name='logistic', tau=[0.45, 0.425])
        if args.model == 'joint_constrained_model':
            dict_file = '../results/save/synthetic_prob_dict_' + str(i) + '.pkl'
            utils.save_dict(train_fold, dict_file)
            prob_dict = utils.load_dict(dict_file)

            tr, te = run_joint_model(logging=logging, train_data=train_fold, test_data=test_fold, 
                                     phi_name='logistic', generate_eval_data=generate_eval_data,
                                     s_name='s', x1_name='x1', y1_name='y1', x2_name='x2', y2_name='y2',
                                     name='synthetic', iters=5000, lr=0.005, lmd1=5.0, lmd2=5.0,
                                     prob_dict=prob_dict, tau=[0.4, 0.955])
        tr_res.append(tr)
        te_res.append(te)

    # ---------- display results -------------------------------
    utils.display_result(tr_res, 'train_data', logging)
    utils.display_result(te_res, 'test_data', logging)

    t_end = time.time()
    logging.info(f"\nTimes: {t_end - t_start:.1f}s")