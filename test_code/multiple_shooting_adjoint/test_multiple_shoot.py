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


hrf_file = 'mt_hrf_estimates.csv'
hrf = pd.read_csv(hrf_file).to_numpy().reshape(-1)#*10.0

N_patients = 1
N_epoch = 50
lr = 1e-2
gamma = 0.95
Lambda = 1.0 # hyper parameter to control weight on grad loss
order =  100 # order for fourier analysis
sparse_lambda = 0.0
plot_test = True
TR = 2.0  # volume interval time in seconds
noise_sigma = 0.1
smooth_penalty = 1.0
repeat_num = 1
case = 3

time_span = np.linspace(0, 200.0, 200)  # 20 orbital periods and 500 points
time_length = time_span.shape[0]
chunk_length = 5 # number of observation points for each chunk, perhaps should change to length of time interval considering irregularly sampled data

save_file_name = 'experiment_1_noise_sigma_{}'.format(noise_sigma)

A_all = []
A_all.append(np.array([
            [0.0, -0.30, -0.1],
            [0.30, 0.0, 0.30],
            [0.10, -0.3, 0.0]
        ]))
A_all.append(np.array([
            [0.0, -0.00, -0.1],
            [0.00, 0.0, 0.30],
            [0.10, -0.3, 0.0]
        ]))
A_all.append(np.array([
            [0.0, -0.00, -0.1],
            [0.00, 0.0, 0.30],
            [0.10, -0.1, 0.0]
        ]))
A_all.append(np.array([
            [0.0, -0.00, -0.1],
            [0.00, 0.0, 0.30],
            [0.10, -0.0, 0.0]
        ]))

A = A_all[case]

A_mask = (np.abs(A)>0).astype(float)

C = np.array( [
            [ 0, 0, 0],
            [ 0, 0, 0]
        ])

C_mask = (np.abs(C)>0).astype(float)

N_ROIs = A.shape[0]

def design_func(t):
    if (math.ceil(t) / 2) % 2 == 0:
        return np.array([1, 0])
    else:
        return np.array([0, 1.0])

class DCMFunc(nn.Module):
    def __init__(self, A, C, A_mask, C_mask):
        super(DCMFunc, self).__init__()
        self.A = nn.Parameter(
            torch.zeros(3,3),
            #torch.from_numpy( A ).float(),
            requires_grad=True )
        self.C = nn.Parameter( torch.from_numpy(C).float(), requires_grad=True )
        self.A_mask = nn.Parameter( torch.from_numpy(A_mask).float(), requires_grad=False )
        self.C_mask = nn.Parameter( torch.from_numpy(C_mask).float(), requires_grad=False )

    def forward(self, t, y):
        design = torch.from_numpy( design_func(t) ).float()
        y_derivative = torch.matmul( y, self.A * self.A_mask) + torch.matmul( design, self.C * self.C_mask)
        return y_derivative

dcmfunc = DCMFunc( A=A, C=C, A_mask=A_mask, C_mask=C_mask )

if not os.path.exists('./results'):
    os.mkdir('results')

save_folder = 'results/task_dynamic_results_Lambda{}_s_lambda{}_order_{}_y41'.format(Lambda, sparse_lambda, order)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

def load_time_series(index): # time_points x ROIs

    def fmri_evolve(w, t):
        return np.dot( w, A) + np.dot(  design_func(t), C )

    # Package initial parameters
    init_params = np.array([0.0, -0.3, 0.2])  # Initial parameters

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

def get_design_func(index):
    # load time series into a  time_points x ROI  shape numpy array
    def design_func_all(t): # input t in a pytorch tensor
        t_all = t.view(-1).data.cpu().numpy().tolist()
        if isinstance(t_all, float):
            t_all = [t_all]
        all = []
        for t in t_all:
            all.append(design_func(t))
        return np.stack(all, 1) # time_points x ROI
    return design_func

results_repeat = []

for _iter in range(repeat_num):

    _start = time()

    # load time series
    data = load_time_series(_iter)
    time_series = data

    if order >= data.shape[-1]:
        order = int(data.shape[-1]) -1
        warnings.warn('We found error when order >= number of time points, please check your code')

    # get design func
    design_func = get_design_func(_iter)

    # generate data
    input_tensor = torch.from_numpy(time_series).float()  # .cuda()  # ROI x time_length
    time_points = time_span.tolist()

    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': 0.1})
    options.update({'rtol': 1e-5})
    options.update({'atol': 1e-6})
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

        print( dcmfunc.A * dcmfunc.A_mask)

    # concatenate by time, and plot
    prediction2, data2 = [], []
    for prediction, data in zip( prediction_chunks, data_chunks ):
        if data.shape[0]>1:
            prediction2.append( prediction[:-1,...] )
            data2.append( data[:-1,...] )

    prediction_all = torch.cat(prediction2, 0)
    data_all = torch.cat( data2, 0 )

fig, axs = plt.subplots(3)
axs[0].plot(prediction_all[:, 0].data.cpu().numpy(), label = 'fitting')
axs[0].plot(data_all[:, 0].data.cpu().numpy(), label = 'data', alpha=0.5)

axs[1].plot(prediction_all[:, 1].data.cpu().numpy(), label = 'fitting')
axs[1].plot(data_all[:, 1].data.cpu().numpy(), label = 'data', alpha=0.5)

axs[2].plot(prediction_all[:, 2].data.cpu().numpy(), label = 'fitting')
axs[2].plot(data_all[:, 2].data.cpu().numpy(), label = 'data', alpha=0.5)

plt.legend()
plt.show()