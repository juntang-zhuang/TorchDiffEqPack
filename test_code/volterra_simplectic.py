""" Lotka volterra DDE """

from pylab import array, linspace, subplots
from TorchDiffEqPack.odesolver import odesolve
import torch.nn as nn
import torch
import scipy as sci
import numpy as np
from matplotlib import pyplot as plt

class Func(nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        self.delay = 0.2
        self.weight = nn.Parameter(torch.ones(1))
    def forward(self, t, Y):
        x, y = Y[0], Y[1]
        xd, yd = x, y
        a1 = self.weight * x * (1 - yd)
        a2 = - self.weight * y * (1 - xd)
        out = torch.cat((a1, a2), 0)
        return out

class History(nn.Module):
    def __init__(self):
        super(History, self).__init__()
        self.w = nn.Parameter(torch.from_numpy(np.array([1,2])).float())
    def forward(self, t):
        return self.w

func = Func()
history = History()

time_span = sci.linspace(0, 10.0, 1000)  # 20 orbital periods and 500 points
t_list = time_span.tolist()

# configure training options
options = {}
options.update({'method': 'sym12async'})
options.update({'t0': 0.0})
options.update({'t1': 10.0})
options.update({'h': None})
options.update({'rtol': 1e-3})
options.update({'atol': 1e-4})
options.update({'print_neval': False})
options.update({'neval_max': 1000000})
options.update({'safety': None})
options.update({'t_eval':t_list})
options.update({'dense_output':True})
options.update({'interpolation_method':'cubic'})


out = odesolve(func, history(0.0), options)
out = out.data.cpu().numpy()
plt.plot(out[:,0], out[:,1])
plt.show()

