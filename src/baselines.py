from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import torch
from Causal_model import *
from utils import *

def evaluate_pred(y_pred, y_true, metrics_set):
    '''
    :param y_pred: numpy array, size = instance num
    :param y_true: numpy array, size = instance num
    :param metrics_set: {'RMSE', 'MAE', 'Acc', ...}
    :return:
    '''
    evaluations = {}
    if 'RMSE' in metrics_set:
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        evaluations['RMSE'] = rmse
    if 'MAE' in metrics_set:
        #mae = np.mean(np.abs(y_true - y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        evaluations['MAE'] = mae

    return evaluations


# constant predictor
def run_constant(x, y, trn_idx, tst_idx):
    y_const = np.mean(y[trn_idx], axis=0)  # the average y in train data
    y_pred_tst = np.full(len(tst_idx), y_const)
    return y_pred_tst

# full predictor
def run_full(x, y, env, trn_idx, tst_idx):
    clf = LinearRegression()  # linear ? logistic ?

    features_full = np.concatenate([x, env], axis=1)
    features_full_trn = features_full[trn_idx]
    features_full_tst = features_full[tst_idx]

    clf.fit(features_full_trn, y[trn_idx])  # train

    # test
    y_pred_tst = clf.predict(features_full_tst)
    return y_pred_tst, clf

# unaware predictor
def run_unaware(x, y, trn_idx, tst_idx):
    clf = LinearRegression()  # linear ? logistic ?
    clf.fit(x[trn_idx], y[trn_idx])  # train

    # test
    y_pred_tst = clf.predict(x[tst_idx])
    return y_pred_tst, clf

# counterfactual fairness predictor with correct causal model
def run_CFP_1_true(x, y, env, trn_idx, tst_idx, model_name, args):
    if args.data == 'law':
        data_trn = {
            'race': env[trn_idx],
            'gpa': x[trn_idx, 1],
            'sat': x[trn_idx, 0].long().float(),
            'fya': y[trn_idx].view(-1)
        }
        data_tst = {
            'race': env[tst_idx],
            'gpa': x[tst_idx, 1],
            'sat': x[tst_idx, 0].long().float(),
            'fya': y[tst_idx].view(-1)
        }

    n_trn = len(trn_idx)
    n_tst = len(tst_idx)

    # training data
    causal_model_trn = get_causal_model(args, data_trn, n_trn, model_name)  # train causal model to fit in the train data
    #test_cause_model(args, causal_model_trn, data_trn, num_samples=200)

    var_unobs_trn = causal_model_trn.get_unobs_var()  # N x d_unobs
    var_unobs_trn = var_unobs_trn.detach().numpy()

    standardize = True
    if standardize:
        var_unobs_trn = standerlize(var_unobs_trn)

    if args.data == 'law':
        x_fair_trn = var_unobs_trn

    #clf = LinearRegression()
    #clf = svm.SVC()
    clf = LinearRegression()
    clf.fit(x_fair_trn, y[trn_idx])  # train

    # test
    causal_model_tst = get_causal_model(args, data_tst, n_tst,
                                                   model_name)  # train causal model to fit in the train data
    var_unobs_tst = causal_model_tst.get_unobs_var()  # N x d_unobs
    var_unobs_tst = var_unobs_tst.detach().numpy()
    if standardize:
        var_unobs_tst = standerlize(var_unobs_tst)

    if args.data == 'law':
        x_fair_tst = var_unobs_tst

    y_pred_tst = clf.predict(x_fair_tst)
    #y_pred_tst = y_pred_tst.reshape(-1)
    return y_pred_tst, clf, causal_model_tst


