import torch
from adabelief_pytorch import AdaBelief
from TorchDiffEqPack import odesolve
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.optim import Adam
from scipy import interpolate
from time import time
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting
from numpy import loadtxt
from scipy.signal import deconvolve, convolve
from scipy.io import loadmat
import  math
import warnings
from scipy import integrate
from multiple_shooting import MultipleShoot
from torch import nn


N_patients = 1
N_epoch = 200
lr = 1e-2
gamma = 0.95
Lambda = 1.0 # hyper parameter to control weight on grad loss
order =  100 # order for fourier analysis
sparse_lambda = 0.0
plot_test = True
TR = 2.0  # volume interval time in seconds
noise_sigma = 0.5
smooth_penalty = 10.0
repeat_num = 1

time_span = np.linspace(0, 160.0, 1600)  # 20 orbital periods and 500 points
time_length = time_span.shape[0]
chunk_length = 10 # number of observation points for each chunk, perhaps should change to length of time interval considering irregularly sampled data

save_file_name = 'lotka'

alpha = 2/3
beta = 4/3
gamma = 1.0
delta = 1.0

class DCMFunc(nn.Module):
    def __init__(self):
        super(DCMFunc, self).__init__()
        self.alpha = nn.Parameter( torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.delta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, t, y):
        out = y * 0.0
        out[0] = self.alpha * y[0] - self.beta * y[0] * y[1]
        out[1] = self.delta * y[0] * y[1] - self.gamma * y[1]
        return out

dcmfunc = DCMFunc( )

def load_time_series(index): # time_points x ROIs

    def fmri_evolve(w, t):
        x , y = w[0], w[1]
        out = w * 0.0
        out[0] = alpha * x - beta * x * y
        out[1] = delta * x * y - gamma * y
        return out

    # Package initial parameters
    init_params = np.array([1.0, 0.2])  # Initial parameters

    # Run the ODE solver
    simulation = integrate.odeint(fmri_evolve, y0=init_params, t=time_span, rtol=1e-5,atol=1e-6)

    '''
    def deconv(x):
        outs = []
        for i in range(x.shape[0]):
            recovered, remainedr = deconvolve(x[i,:], hrf)
            outs.append(recovered)
        outs = np.stack(outs, axis=0)
        return outs
    '''

    outs = simulation + np.random.rand( simulation.shape[0], simulation.shape[1] ) * noise_sigma
    return outs

results_repeat = []

for _iter in range(repeat_num):

    _start = time()

    # load time series
    data = load_time_series(_iter)
    time_series = data

    if order >= data.shape[-1]:
        order = int(data.shape[-1]) -1
        warnings.warn('We found error when order >= number of time points, please check your code')

    # generate data
    input_tensor = torch.from_numpy(time_series).float()  # .cuda()  # ROI x time_length
    time_points = time_span.tolist()

    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': 0.1})
    options.update({'rtol': 1e-7})
    options.update({'atol': 1e-7})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    # create multiple-shooting instance
    multi_shoot = MultipleShoot( ode_func=dcmfunc, observation_length=time_length, ODE_options=options,
                                 smooth_penalty=smooth_penalty, chunk_length=chunk_length)
    multi_shoot.prepare_intermediate( input_tensor )

    # create model
    optimizer = AdaBelief(filter(lambda p: p.requires_grad, multi_shoot.parameters()), lr = lr, eps=1e-16, rectify=False,
                          betas=(0.5, 0.9))
    #optimizer = Adam(filter(lambda p: p.requires_grad, multi_shoot.parameters()), lr=lr, eps=1e-16,
    #                      betas=(0.5, 0.9))

    for _epoch in range(N_epoch):
        # adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma

        optimizer.zero_grad()

        # forward and optimizer
        prediction_chunks, data_chunks = multi_shoot.fit_and_grad( input_tensor, time_points )
        loss = multi_shoot.get_loss( prediction_chunks, data_chunks )

        loss.backward(retain_graph=False)
        optimizer.step()

        print( 'Alpha {}, Beta {}, Gamma {}, Delta {}'.format(dcmfunc.alpha, dcmfunc.beta, dcmfunc.gamma, dcmfunc.delta))

    # concatenate by time, and plot
    prediction2, data2 = [], []
    for prediction, data in zip( prediction_chunks, data_chunks ):
        if data.shape[0]>1:
            prediction2.append( prediction[:-1,...] )
            data2.append( data[:-1,...] )

    prediction_all = torch.cat(prediction2, 0)
    data_all = torch.cat( data2, 0 )

fig, axs = plt.subplots(2)
axs[0].plot(prediction_all[:, 0].data.cpu().numpy(), label = 'fitting')
axs[0].plot(data_all[:, 0].data.cpu().numpy(), label = 'data', alpha=0.5)

axs[1].plot(prediction_all[:, 1].data.cpu().numpy(), label = 'fitting')
axs[1].plot(data_all[:, 1].data.cpu().numpy(), label = 'data', alpha=0.5)

plt.legend()
plt.show()
