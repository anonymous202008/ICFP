import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import pandas as pd
import csv
import random
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0")

def standerlize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 0.000001)
    return x

def evaluate_fairness(y_cf1, y_cf2):  # y_cf1: n x samplenum
    mmd = mmd2_lin(y_cf1, y_cf2, 0.3)

    wass, _ = wasserstein(y_cf1, y_cf2, device, cuda=True)

    return mmd, wass

def data_statistics(path, name):
    print('checking dataset: ', name)
    if name == 'law':
        csv_data = pd.read_csv(path)
        print("instance num and feature num: ", csv_data.shape)

        # sensitive attributes
        race = csv_data['race'].value_counts()
        print(race)

        sex = csv_data['sex'].value_counts()
        print(sex)
    if name == 'adult':
        csv_data = pd.read_csv(path)
        print("instance num and feature num: ", csv_data.shape)

        # sensitive attributes
        race = csv_data['race'].value_counts()
        print(race)

        sex = csv_data['sex'].value_counts()
        print(sex)
    return

def load_data(path, name):
    print('loading dataset: ', name)
    if name == 'law':
        csv_data = pd.read_csv(path)

        races = ['White', 'Black', 'Asian', 'Hispanic', 'Mexican', 'Other', 'Puertorican', 'Amerindian']
        sexes = [1, 2]

        # index selection
        selected_races = ['White', 'Black', 'Asian']
        print("select races: ", selected_races)
        select_index = np.array(csv_data[(csv_data['race'] == selected_races[0]) | (csv_data['race'] == selected_races[1]) |
                                         (csv_data['race'] == selected_races[2])].index, dtype=int)
        # shuffle
        np.random.shuffle(select_index)

        x = csv_data[['LSAT','UGPA']].to_numpy()[select_index]  # n x d
        y = csv_data[['ZFYA']].to_numpy()[select_index]  # n x 1

        n = x.shape[0]
        env_race = csv_data['race'][select_index].to_list()  # n, string list
        env_race_onehot = np.zeros((n, len(selected_races)))
        #env_sex = csv_data['sex'][select_index].to_list()  # n, int list
        #env_sex_onehot = np.zeros((n, len(sexes)))
        for i in range(n):
            env_race_onehot[i][selected_races.index(env_race[i])] = 1.   # n x len(selected_races)
            #env_sex_onehot[i][sexes.index(env_sex[i])] = 1.
        #env = np.concatenate([env_race_onehot, env_sex_onehot], axis=1)  # n x (No. races + No. sex)
        env = env_race_onehot

    return x, y, env

def load_data_old(path, name):
    if name == 'law':
        print('loading dataset: ', name)
        csv_data = pd.read_csv(path)

        races = ['White', 'Black', 'Asian', 'Hispanic', 'Mexican', 'Other', 'Puertorican', 'Amerindian']
        sexes = [1, 2]

        # index selection
        selected_races = ['White', 'Black']
        select_index = np.array(csv_data[(csv_data['race'] == selected_races[0]) | (csv_data['race'] == selected_races[1])].index, dtype=int)
        # shuffle
        np.random.shuffle(select_index)

        x = csv_data[['LSAT','UGPA']].to_numpy()[select_index]  # n x d
        y = csv_data[['ZFYA']].to_numpy()[select_index]  # n x 1

        n = x.shape[0]
        env_race = csv_data['race'][select_index].to_list()  # n, string list
        env_race_onehot = np.zeros((n, len(selected_races)))
        env_sex = csv_data['sex'][select_index].to_list()  # n, int list
        env_sex_onehot = np.zeros((n, len(sexes)))
        for i in range(n):
            env_race_onehot[i][selected_races.index(env_race[i])] = 1.
            env_sex_onehot[i][sexes.index(env_sex[i])] = 1.
        env = np.concatenate([env_race_onehot, env_sex_onehot], axis=1)

        # plot
        flag_plot = False
        if flag_plot:
            font_sz = 20
            fig, ax = plt.subplots(nrows=2, ncols= 2 * len(selected_races), figsize=(12, 6), sharey=True)

            for race_idx in range(len(selected_races)):
                race_cur = selected_races[race_idx]
                data_female = csv_data[(csv_data["race"] == race_cur) & (csv_data['sex'] == 1)]
                data_male = csv_data[(csv_data["race"] == race_cur) & (csv_data['sex'] == 2)]

                ax[race_idx][0].scatter(data_male['LSAT'],
                                data_male['ZFYA'], 3, marker='o', label=race_cur+'male', color='black')
                ax[race_idx][1].scatter(data_male['UGPA'],
                                data_male['ZFYA'], 5, marker='o', label=race_cur+'male', color='blue')
                ax[race_idx][2].scatter(data_female['LSAT'],
                                         data_female['ZFYA'], 3, marker='o', label=race_cur + 'female', color='red')
                ax[race_idx][3].scatter(data_female['UGPA'],
                                             data_female['ZFYA'], 5, marker='o', label=race_cur + 'female', color='yellow')

                ax[race_idx][0].set(xlabel="LSAT", ylabel="FYA", title=race_cur+'male')
                ax[race_idx][1].set(xlabel="UGPA", ylabel="FYA", title=race_cur+'male')
                ax[race_idx][2].set(xlabel="LSAT", ylabel="FYA", title=race_cur + 'female')
                ax[race_idx][3].set(xlabel="UGPA", ylabel="FYA", title=race_cur + 'female')

            plt.show()
    return x, y, env

def evaluate_fair_with_index(y_cf_list, idx_e):
    compare_num = 0
    mmd_list = []
    wass_list = []
    for i in range(len(y_cf_list)):
        for j in range(len(y_cf_list)):
            if i == j:
                continue

            compare_num += 1
            mmd, wass = evaluate_fairness(y_cf_list[i][idx_e], y_cf_list[j][idx_e])
            mmd_list.append(mmd)
            wass_list.append(wass)

    ave_mmd = sum(mmd_list) / len(mmd_list)
    ave_wass = sum(wass_list) / len(wass_list)

    return ave_mmd, ave_wass

def get_index_for_env(env_cur, distinct_envs, args):
    idx_each_env = []  # idx_each_env[i]: index of those "env = distinct_envs[i]" in env_cur
    for e in distinct_envs:
        idx_e = []
        if args.cuda:
            e=e.to(device)
        for i in range(env_cur.shape[0]):
            if torch.equal(env_cur[i], e):
                idx_e.append(i)

        idx_e = torch.LongTensor(idx_e)
        if args.cuda:
            idx_e = idx_e.to(device)

        idx_each_env.append(idx_e)
    return idx_each_env

def make_environments(x, y, env, args):
    '''
    separate the data by different environments
    :param x: tensor, shape: n x d
    :param y: tensor, shape: n x 1
    :param env: tensor, shape: n x d_e
    :return: a list, size = Num of environments, each elem is a dict {x, y, env}
    '''
    env_data = []
    distinct_envs = torch.unique(env, dim=0)  # unique rows
    for e in distinct_envs:
        #idx_e = torch.where(env == e)
        idx_e = []
        for i in range(env.shape[0]):
            if torch.equal(env[i], e):
                idx_e.append(i)

        idx_e = torch.LongTensor(idx_e)
        if args.cuda:
            idx_e = idx_e.to(device)

        env_data.append({
            'x': x[idx_e],
            'y': y[idx_e],
            'env': env[idx_e],
            'orin_index': idx_e
        })
    return env_data

def split_trn_tst_random(trn_rate, tst_rate, n):
    trn_id_list = random.sample(range(n), int(n * trn_rate))
    not_trn = list(set(range(n)) - set(trn_id_list))
    tst_id_list = random.sample(not_trn, int(n * tst_rate))
    val_id_list = list(set(not_trn) - set(tst_id_list))
    trn_id_list.sort()
    val_id_list.sort()
    tst_id_list.sort()
    return trn_id_list, val_id_list, tst_id_list

def mean_nll(logits, y):
    #return nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss_mse = nn.MSELoss(reduction='mean').to(device)
    return loss_mse(logits.view(-1), y.view(-1))

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def mmd2_lin(Xt, Xc,p):
    ''' Linear MMD '''
    mean_control = torch.mean(Xc,0)
    mean_treated = torch.mean(Xt,0)

    mmd = torch.sum((2.0*p*mean_treated - 2.0*(1.0-p)*mean_control) ** 2)

    return mmd


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)



def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]

    #x = x.squeeze()
    #y = y.squeeze()

    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam / M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape).to(device)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape).to(device), torch.zeros((1, 1)).to(device)], 0)
    if cuda:
        #row = row.cuda()
        #col = col.cuda()
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        #temp_term = temp_term.cuda()
        #a = a.cuda()
        #b = b.cuda()
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            #u = u.cuda()
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        #v = v.cuda()
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        #D = D.cuda()
        D = D.to(device)

    return D, Mlam

def safe_sqrt(x, lbound=1e-10):
    ''' Numerically safe version of pytorch sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, np.inf))