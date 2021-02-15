import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

from torch import nn
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.poutine.trace_messenger import TraceMessenger

pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


#plt.style.use('default')

class CausalModel(PyroModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, data):
        if self.model_name == 'law_l2':
            dim_race = 3
            self.b_g = pyro.param(self.model_name + "_" + "b_g", torch.tensor(0.))
            self.w_g_k = pyro.param(self.model_name + "_" + "w_g_k", torch.tensor(0.))
            self.w_g_r = pyro.param(self.model_name + "_" + "w_g_r", torch.zeros(dim_race, 1))
            self.sigma_g = pyro.param(self.model_name + "_" + "sigma_g", torch.tensor(1.))

            self.b_l = pyro.param(self.model_name + "_" + "b_l", torch.tensor(0.))
            self.w_l_k = pyro.param(self.model_name + "_" + "w_l_k", torch.tensor(0.))
            self.w_l_r = pyro.param(self.model_name + "_" + "w_l_r", torch.zeros(dim_race, 1))

            self.w_f_k = pyro.param(self.model_name + "_" + "w_f_k", torch.tensor(0.))
            self.w_f_r = pyro.param(self.model_name + "_" + "w_f_r", torch.zeros(dim_race, 1))

            with pyro.plate(self.model_name + "_data", data['fya'].shape[0]):
                knowledge_loc = data['fya'].new_zeros(torch.Size((data['fya'].shape[0], 1)))  # knowledge: N X 1
                knowledge_scale = data['fya'].new_ones(torch.Size((data['fya'].shape[0], 1)))

                self.knowledge = pyro.sample(self.model_name + "_" + "knowledge",
                                        dist.Normal(knowledge_loc, knowledge_scale).to_event(1))

                gpa_mean = self.b_g + self.w_g_k * self.knowledge + data['race'] @ self.w_g_r
                #sat_mean = torch.exp(self.b_l + self.w_l_k * self.knowledge + data['race'] @ self.w_l_r)
                sat_mean = self.b_l + self.w_l_k * self.knowledge + data['race'] @ self.w_l_r
                fya_mean = self.w_f_k * self.knowledge + data['race'] @ self.w_f_r

                pred_mean = torch.cat([gpa_mean, sat_mean, fya_mean], dim=1)  # N x 3

                gpa_obs = pyro.sample(self.model_name+"_"+"gpa", dist.Normal(gpa_mean, torch.abs(self.sigma_g)).to_event(1), obs=data['gpa'].reshape(-1, 1))
                #sat_obs = pyro.sample(self.model_name+"_"+"sat", dist.Poisson(sat_mean).to_event(1), obs=data['sat'].reshape(-1, 1))
                sat_obs = pyro.sample("sat", dist.Normal(sat_mean, 1).to_event(1), obs=data['sat'].reshape(-1, 1))
                fya_obs = pyro.sample(self.model_name+"_"+"fya", dist.Normal(fya_mean, 1).to_event(1), obs=data['fya'].reshape(-1, 1))

            # return pred_mean

        elif self.model_name == 'law_false':
            dim_race = 3
            # self.b_g = pyro.param(self.model_name + "_" + "b_g", torch.tensor(0.))
            # self.w_g_k = pyro.param(self.model_name + "_" + "w_g_k", torch.tensor(0.))
            # self.w_g_r = pyro.param(self.model_name + "_" + "w_g_r", torch.zeros(dim_race, 1))
            # self.sigma_g = pyro.param(self.model_name + "_" + "sigma_g", torch.tensor(1.))

            self.b_l = pyro.param(self.model_name + "_" + "b_l", torch.tensor(0.))
            self.w_l_k = pyro.param(self.model_name + "_" + "w_l_k", torch.tensor(0.))
            self.w_l_r = pyro.param(self.model_name + "_" + "w_l_r", torch.zeros(dim_race, 1))

            self.w_f_k = pyro.param(self.model_name + "_" + "w_f_k", torch.tensor(0.))
            self.w_f_r = pyro.param(self.model_name + "_" + "w_f_r", torch.zeros(dim_race, 1))

            with pyro.plate(self.model_name + "_data", data['fya'].shape[0]):
                knowledge_loc = data['fya'].new_zeros(torch.Size((data['fya'].shape[0], 1)))
                knowledge_scale = data['fya'].new_ones(torch.Size((data['fya'].shape[0], 1)))

                self.knowledge = pyro.sample(self.model_name + "_" + "knowledge",
                                        dist.Normal(knowledge_loc, knowledge_scale).to_event(1))

                #gpa_mean = self.b_g + self.w_g_k * self.knowledge + data['race'] @ self.w_g_r
                #sat_mean = torch.exp(self.b_l + self.w_l_k * self.knowledge + data['race'] @ self.w_l_r)
                sat_mean = self.b_l + self.w_l_k * self.knowledge + data['race'] @ self.w_l_r
                fya_mean = self.w_f_k * self.knowledge + data['race'] @ self.w_f_r

                #pred_mean = torch.cat([gpa_mean, sat_mean, fya_mean], dim=1)

                #gpa_obs = pyro.sample(self.model_name+"_"+"gpa", dist.Normal(gpa_mean, torch.abs(self.sigma_g)).to_event(1), obs=data['gpa'].reshape(-1, 1))
                #sat_obs = pyro.sample(self.model_name+"_"+"sat", dist.Poisson(sat_mean).to_event(1), obs=data['sat'].reshape(-1, 1))
                sat_obs = pyro.sample("sat", dist.Normal(sat_mean, 1).to_event(1), obs=data['sat'].reshape(-1, 1))
                fya_obs = pyro.sample(self.model_name+"_"+"fya", dist.Normal(fya_mean, 1).to_event(1), obs=data['fya'].reshape(-1, 1))

    def guide(self, data): # q(z|x)
        with pyro.plate(self.model_name +"_data", data['fya'].shape[0]):
            self.knowledge_loc = pyro.param(self.model_name+"_"+"knowledge_loc", data['fya'].new_zeros(torch.Size((data['fya'].shape[0], 1))))  # knowledge: N X 1
            self.knowledge_scale = pyro.param(self.model_name+"_"+"knowledge_scale", data['fya'].new_ones(torch.Size((data['fya'].shape[0], 1))))

            pyro.sample(self.model_name + "_" + "knowledge", dist.Normal(self.knowledge_loc, self.knowledge_scale).to_event(1))


    def get_unobs_var(self):
        if self.model_name == 'law_l2' or self.model_name == 'law_false':
            #return self.knowledge
            return self.knowledge_loc

    def get_unobs_var_samples(self):
        if self.model_name == 'law_l2' or self.model_name == 'law_false':
            #return self.knowledge
            knowledge_samples = pyro.sample("knowledge_samples_samples", dist.Normal(self.knowledge_loc, self.knowledge_scale))
            return knowledge_samples

def get_causal_model(args, data, N, model_name):
    dataset = {}
    num_iterations = args.num_iterations_cm

    if model_name == 'law_l2' or model_name == 'law_false':
        cm_model = CausalModel(model_name)

        adam = pyro.optim.Adam({"lr": args.lr_cm})
        svi = SVI(cm_model.forward, cm_model.guide, adam, loss=Trace_ELBO())

        pyro.clear_param_store()
        # training
        for j in range(num_iterations):
            # calculate the loss and take a gradient step
            loss = svi.step(data)
            if j % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data['fya'])))

    return cm_model


def get_data_cf(args, causal_model, data, attr_do, num_samples=500):
    data_cf_y_all = None
    data_cf_x_all = None
    data_cf_env = attr_do['race']  # n x No.races
    data_cf_u_all = None

    if args.data == 'law':
        for i in range(num_samples):
            knowledge_samples = causal_model.get_unobs_var_samples()  # N x 1

            data_cf_gpa = causal_model.b_g + causal_model.w_g_k * knowledge_samples + data['race'] @ causal_model.w_g_r  # N X 1
            # sat_mean = torch.exp(self.b_l + self.w_l_k * knowledge_samples + data['race'] @ self.w_l_r)
            data_cf_sat = causal_model.b_l + causal_model.w_l_k * knowledge_samples + data['race'] @ causal_model.w_l_r

            data_cf_y = causal_model.w_f_k * knowledge_samples + data['race'] @ causal_model.w_f_r
            data_cf_x = torch.cat([data_cf_sat, data_cf_gpa], dim=1)  # sample x n x dim_x

            if data_cf_y_all is None:
                data_cf_y_all = data_cf_y.unsqueeze(0)
            else:
                data_cf_y_all = torch.cat([data_cf_y_all, data_cf_y.unsqueeze(0)], dim=0)

            if data_cf_x_all is None:
                data_cf_x_all = data_cf_x.unsqueeze(0)
            else:
                data_cf_x_all = torch.cat([data_cf_x_all, data_cf_x.unsqueeze(0)], dim=0)

            if data_cf_u_all is None:
                data_cf_u_all = knowledge_samples.unsqueeze(0)
            else:
                data_cf_u_all = torch.cat([data_cf_u_all, knowledge_samples.unsqueeze(0)], dim=0)

        data_cf = {
            'x': data_cf_x_all.cpu().detach().numpy(),
            'y': data_cf_y_all.cpu().detach().numpy(),
            'env': data_cf_env.cpu().detach().numpy(),
            'u': data_cf_u_all.cpu().detach().numpy()  # samples x n x 1
        }

        # to numpy

        return data_cf



def test_cause_model(args, cm_model, guide, data, num_samples=200):
    if args.data == 'law':
        predictive = Predictive(cm_model, guide=guide, num_samples=num_samples,
                                return_sites=("gpa", "sat", "fya", "_RETURN"))
        samples = predictive(data)

        pred_summary = summary(samples)

        pred_mean = pred_summary["_RETURN"]


