import time
import argparse
import numpy as np
import pandas as pd
import csv
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from IRMmodel import IRMmodel
from Causal_model import *
import matplotlib.pyplot as plt
from baselines import *
import copy
from utils import *

parser = argparse.ArgumentParser(description='Causal fairness through IRM')
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--data', default='law', help='Dataset name')  # 'law', 'adult'
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_iterations_cm', type=int, default=1000, metavar='N',
                    help='number of epochs to train causal model (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--h_dim', type=int, default=5, metavar='N',
                    help='dimension of hidden variables')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate for optimizer')
parser.add_argument('--lr_cm', type=float, default=0.001,
                    help='learning rate for causal mode')
parser.add_argument('--l2_regularizer_weight', type=float, default=1e-3,
                    help='l2_regularizer_weight')

args = parser.parse_args()

# select gpu if available
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
args.device = device

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def train(env_data_trn, model, optimizer, args):
    '''
    :param env_data_trn: a list, size = Num of environments, each elem is a dict {x, y, env}
    :param model:
    :param optimizer:
    :param args:
    :return:
    '''
    time_begin = time.time()

    # invariant
    penalty_weight = 1.0
    penalty_anneal_iters = 100

    model.train()
    print("start training!")

    loss_mse = nn.MSELoss(reduction='mean').to(device)

    for epoch in range(args.epochs):
        loss = torch.tensor(0.).to(device)
        loss_y = torch.tensor(0.).to(device)
        loss_irm = torch.tensor(0.).to(device)
        optimizer.zero_grad()

        rep_list = []

        for data_e in env_data_trn:
            x_e = data_e['x']
            y_e = data_e['y']
            env_e = data_e['env']

            y_pred, rep_e = model(x_e, env_e)

            # invariant
            penalty_e = penalty(y_pred, y_e)
            penalty_weight = (penalty_weight
                              if epoch >= penalty_anneal_iters else 1.0)
            loss_irm += penalty_weight * penalty_e  # IRM term


            rep_list.append(rep_e)

        x_all = torch.cat([env_data_trn[0]['x'], env_data_trn[1]['x'], env_data_trn[2]['x']], dim=0)
        y_all = torch.cat([env_data_trn[0]['y'], env_data_trn[1]['y'], env_data_trn[2]['y']], dim=0)
        env_all = torch.cat([env_data_trn[0]['env'], env_data_trn[1]['env'], env_data_trn[2]['env']], dim=0)

        y_pred, rep_e = model(x_all, env_all)
        loss_y = loss_mse(y_pred.view(-1), y_all.view(-1))
        loss += loss_y

        # irm
        loss += loss_irm / len(env_data_trn)

        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss += args.l2_regularizer_weight * weight_norm  # l2_regularization

        # invariant
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        # rep dist
        rep_dist = torch.tensor(0.).to(device)
        num_ijs = 0
        for i in range(len(env_data_trn)):
            for j in range(len(env_data_trn)):
                if i == j:
                    continue
                rep_dist_ij = mmd2_lin(rep_list[i], rep_list[j], 0.3)
                rep_dist += rep_dist_ij
                num_ijs += 1
        rep_dist /= num_ijs
        #rep_dist = wasserstein(rep_list[0], rep_list[1], device, cuda=True)
        # rep_dist = mmd2_lin(rep_list[0], rep_list[1], 0.3)
        mmd_loss = safe_sqrt(0.001 ** 2 * rep_dist)
        #
        loss += 10*mmd_loss  # MMD loss

        # backward propagation
        loss.backward()
        optimizer.step()

        # evaluate
        if epoch % 100 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'loss_y: {:.4f}'.format(loss_y.item()),
                  'loss_irm: {:.4f}'.format(loss_irm.item()),
                  'loss_w2: {:.4f}'.format(weight_norm.item()),
                  'loss_mmd: {:.4f}'.format(mmd_loss.item()))
    return

def test(env_data_tst, model):
    model.eval()

    loss_mse = nn.MSELoss(reduction='mean').to(device)
    loss = torch.tensor(0.).to(device)
    abs_err = 0.0
    sqr_err = 0.0
    num_of_instance = 0

    for data_e in env_data_tst:
        x_e = data_e['x']
        y_e = data_e['y']
        env_e = data_e['env']

        y_pred, rep = model(x_e, env_e)

        num_of_instance += len(y_e)

        loss_e = loss_mse(y_pred.view(-1), y_e.view(-1))
        loss += loss_e
        abs_err += torch.sum(torch.abs(y_pred.view(-1) - y_e.view(-1)))
        sqr_err += torch.sum(torch.square(y_pred.view(-1) - y_e.view(-1)))

    loss /= len(env_data_tst)

    abs_err /= num_of_instance
    sqr_err /= num_of_instance

    print('loss_mse_test: {:.4f}'.format(loss.item()))
    print('mse_test: {:.4f}'.format(sqr_err.item()))
    print('rmse_test: {:.4f}'.format(safe_sqrt(sqr_err).item()))
    print('mae_test: {:.4f}'.format(abs_err.item()))

    return

def predict(data, model):
    model.eval()

    x_e = data['x']
    env_e = data['env']
    y_pred, rep = model(x_e, env_e)

    return y_pred


if __name__ == '__main__':
    trn_rate = 0.6
    tst_rate = 0.2
    model_name = 'law_l2'
    model_name_assume = 'law_l2'

    if args.data == 'law':
        path = '../../dataset/law_data.csv'

    data_statistics(path, args.data)

    x, y, env = load_data(path, args.data)  # environment selection
    x_dim = x.shape[1]
    n = len(x)

    standardize = True
    if standardize:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 0.000001)

    trn_idx, val_idx, tst_idx = split_trn_tst_random(trn_rate, tst_rate, n)

    metrics_set = set({"RMSE", "MAE"})

    y_pred_tst_constant = run_constant(x, y, trn_idx, tst_idx)
    eval_constant = evaluate_pred(y_pred_tst_constant, y[tst_idx], metrics_set)
    print("=========== evaluation for constant predictor ================: ", eval_constant)

    y_pred_tst_full, clf_full = run_full(x, y, env, trn_idx, tst_idx)
    eval_full = evaluate_pred(y_pred_tst_full, y[tst_idx], metrics_set)
    print("=========== evaluation for full predictor ================: ", eval_full)

    y_pred_tst_unaware, clf_unaware = run_unaware(x, y, trn_idx, tst_idx)
    eval_unaware = evaluate_pred(y_pred_tst_unaware, y[tst_idx], metrics_set)
    print("=========== evaluation for unaware predictor ================: ", eval_unaware)

    # numpy -> tensors
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    env = torch.FloatTensor(env)

    if args.data == 'law':
        data_tst = {
            'race': env[tst_idx],
            'gpa': x[tst_idx, 1],
            'sat': x[tst_idx, 0].long().float(),
            'fya': y[tst_idx].view(-1)
        }

    # if args.cuda:
    #     x = x.cuda()
    #     y = y.cuda()
    #     env = env.cuda()

    y_pred_tst_cfp1, clf_cfp1, causal_model_tst = run_CFP_1_true(x, y, env, trn_idx, tst_idx, model_name_assume, args)
    eval_cf_1_true = evaluate_pred(y_pred_tst_cfp1, y[tst_idx], metrics_set)
    print("=========== evaluation for CFPred_1 predictor with true causal model ================: ", eval_cf_1_true)


    model_IRM = IRMmodel(x_dim, args)  # model

    if args.cuda:
        model_IRM = model_IRM.to(device)
        x = x.to(device)
        y = y.to(device)
        env = env.to(device)
    env_data = make_environments(x, y, env, args)  # divide the dataset with environment

    optimizer = optim.Adam(model_IRM.parameters(), lr=args.lr)

    env_cur = env[tst_idx]
    distinct_envs = torch.unique(data_tst['race'], dim=0)
    idx_each_env = get_index_for_env(env_cur, distinct_envs,args)


    env_data_trn = make_environments(x[trn_idx],y[trn_idx],env[trn_idx], args)
    env_data_tst = make_environments(x[tst_idx], y[tst_idx], env[tst_idx], args)

    train_size = [data_e['x'].shape[0] for data_e in env_data_trn]
    print('train: ', ' num of environments: ', len(env_data_trn), ' total instances: ', sum(train_size),
          ' each environment: ', train_size)
    test_size = [data_e['x'].shape[0] for data_e in env_data_tst]
    print('test: ', ' num of environments: ', len(env_data_tst), ' total instances: ', sum(test_size),
          ' each environment: ', test_size)

    instance_num_all = sum(train_size) + sum(test_size)

    # ours: train IRM model to capture the invariant features
    train(env_data_trn, model_IRM, optimizer, args)  # train on some environments
    test(env_data_tst, model_IRM)

    n_tst = len(tst_idx)

    # causal model on test (true)
    causal_model_tst = get_causal_model(args, data_tst, n_tst, model_name)

    unobs = causal_model_tst.get_unobs_var()  # N x d_unobs
    unobs = unobs.detach().numpy()

    data_cf_list = []
    for e in distinct_envs:
        print("==================   counterfacutal ", e, "=======================")
        num_samples_cf = 1000
        attr_do = {"race": e.repeat(n_tst, 1)}
        data_cf = get_data_cf(args, causal_model_tst, data_tst, attr_do=attr_do,
                              num_samples=num_samples_cf)  # sample x N
        data_cf_list.append(data_cf)

    #
    y_cf_list_full = [None for i in range(len(data_cf_list))]
    y_cf_list_unaware = [None for i in range(len(data_cf_list))]
    y_cf_list_cfpred1 = [None for i in range(len(data_cf_list))]
    y_cf_list_irm = [None for i in range(len(data_cf_list))]

    select_samples = 1000

    for i in range(len(data_cf_list)):  # sensitive subgroup
        env_cf = data_cf_list[i]['env']  # n_tst x No.races

        for si in range(select_samples):  # sample id
            x_cf = data_cf_list[i]['x'][si]  # n_tst x dim_x
            y_cf = data_cf_list[i]['y'][si]  # n_tst x 1
            u_cf = data_cf_list[i]['u'][si]
            y_pred_cf_cfp1 = clf_cfp1.predict(u_cf)

            features_full = np.concatenate([x_cf, env_cf], axis=1)
            y_pred_cf_full = clf_full.predict(features_full)  # n_tst x 1
            y_pred_cf_unaware = clf_unaware.predict(x_cf)

            y_pred_cf_cfp1 = torch.FloatTensor(y_pred_cf_cfp1)
            y_pred_cf_full = torch.FloatTensor(y_pred_cf_full)
            y_pred_cf_unaware = torch.FloatTensor(y_pred_cf_unaware)

            if args.cuda:
                y_pred_cf_cfp1 = y_pred_cf_cfp1.to(device)
                y_pred_cf_full = y_pred_cf_full.to(device)
                y_pred_cf_unaware = y_pred_cf_unaware.to(device)

                # IRM
                x_cf = torch.FloatTensor(x_cf)
                y_cf = torch.FloatTensor(y_cf)
                env_cf = torch.FloatTensor(env_cf)

                if args.cuda:
                    x_cf = x_cf.to(device)
                    y_cf = y_cf.to(device)
                    env_cf = env_cf.to(device)

                data_cf_clean = {
                    'x': x_cf,
                    'y': y_cf,
                    'env': env_cf
                }
                y_pred_cf_irm = predict(data_cf_clean, model_IRM)  # N

                env_cf = env_cf.cpu().detach().numpy()

            if y_cf_list_full[i] is None:
                y_cf_list_full[i] = y_pred_cf_full
            else:
                y_cf_list_full[i] = torch.cat([y_cf_list_full[i], y_pred_cf_full], dim=1)

            if y_cf_list_unaware[i] is None:
                y_cf_list_unaware[i] = y_pred_cf_unaware
            else:
                y_cf_list_unaware[i] = torch.cat([y_cf_list_unaware[i], y_pred_cf_unaware], dim=1)

            if y_cf_list_cfpred1[i] is None:
                y_cf_list_cfpred1[i] = y_pred_cf_cfp1
            else:
                y_cf_list_cfpred1[i] = torch.cat([y_cf_list_cfpred1[i], y_pred_cf_cfp1], dim=1)

            if y_cf_list_irm[i] is None:
                y_cf_list_irm[i] = y_pred_cf_irm
            else:
                y_cf_list_irm[i] = torch.cat([y_cf_list_irm[i], y_pred_cf_irm], dim=1)


    mmd_full = []
    mmd_unaware = []
    mmd_cfp1 = []
    mmd_irm = []
    wass_full = []
    wass_unaware = []
    wass_cfp1 = []
    wass_irm = []

    # fairness
    compare_num = 0

    for i in range(len(data_cf_list)):
        for j in range(len(data_cf_list)):
            if i == j:
                continue

            compare_num += 1
            mmd, wass = evaluate_fairness(y_cf_list_full[i], y_cf_list_full[j])
            mmd_full.append(mmd)
            wass_full.append(wass)

            mmd, wass = evaluate_fairness(y_cf_list_unaware[i], y_cf_list_unaware[j])
            mmd_unaware.append(mmd)
            wass_unaware.append(wass)

            mmd, wass = evaluate_fairness(y_cf_list_cfpred1[i], y_cf_list_cfpred1[j])
            mmd_cfp1.append(mmd)
            wass_cfp1.append(wass)

            mmd, wass = evaluate_fairness(y_cf_list_irm[i], y_cf_list_irm[j])
            mmd_irm.append(mmd)
            wass_irm.append(wass)

    ave_mmd_full = sum(mmd_full) / len(mmd_full)
    ave_mmd_unaware = sum(mmd_unaware) / len(mmd_unaware)
    ave_mmd_cfp1 = sum(mmd_cfp1) / len(mmd_cfp1)
    ave_mmd_irm = sum(mmd_irm) / len(mmd_irm)

    ave_wass_full = sum(wass_full) / len(wass_full)
    ave_wass_unaware = sum(wass_unaware) / len(wass_unaware)
    ave_wass_cfp1 = sum(wass_cfp1) / len(wass_cfp1)
    ave_wass_irm = sum(wass_irm) / len(wass_irm)

    print("====================================== overall fairness ==================================================================")

    print("================== fairness evaluation for full prediction ==================: mmd: ", ave_mmd_full, " wass: ", ave_wass_full)
    print("================== fairness evaluation for unaware prediction ==================: mmd: ", ave_mmd_unaware,
          " wass: ", ave_wass_unaware)
    print("================== fairness evaluation for cfp1 prediction ==================: mmd: ", ave_mmd_cfp1,
          " wass: ", ave_wass_cfp1)
    print("================== fairness evaluation for irm-m prediction ==================: mmd: ", ave_mmd_irm,
          " wass: ", ave_wass_irm)
