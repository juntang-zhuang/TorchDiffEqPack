import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from TorchDiffEqPack import  odesolve

t_list = np.linspace(0,3000, 2000)

# configure training options
options = {}
options.update({'method': 'ode23s'})
options.update({'h': None})
options.update({'t0': 0.0})
options.update({'t1': 3000.0})
options.update({'rtol': 1e-5})
options.update({'atol': 1e-4})
options.update({'print_neval': False})
options.update({'neval_max': 1000000})
options.update({'safety': None})
options.update({'t_eval':t_list})
options.update({'print_time':True})

class Func(nn.Module):
    def __init__(self):
        super(Func, self).__init__()
    def forward(self, t, y):
        return torch.stack( [ y[:,1], 1000 * (1 - y[:,0]**2) * y[:,1]  - y[:,0]    ] ,-1)

func = Func()
initial_condition = torch.tensor([[2.0, 0.0]]).float()

out = odesolve(func, initial_condition, options=options)
plt.plot(out[:,0,0].data.cpu().numpy(),'-o')
plt.show()

# class Func(nn.Module):
#     def __init__(self):
#         super(Func, self).__init__()
#     def forward(self, t, y):
#         return y**2 * (1.0 - y)
#
# func = Func()
# initial_condition = torch.tensor([[1e-2]]).float()
#
# out = odesolve(func, initial_condition, options=options)
# plt.plot(t_list, out[:,0].data.cpu().numpy(),'-o')
# plt.show()