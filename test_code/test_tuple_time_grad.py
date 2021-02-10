import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from TorchDiffEqPack import  odesolve

t_list = np.linspace(0,1.0, 100)

class Func(nn.Module):
    def __init__(self):
        super(Func, self).__init__()
    def forward(self, t, y):
        return ( y[1] * t, 10.0 * (1 - y[0]**2) * y[1]  - y[0]  )

func = Func()
initial_condition = (torch.tensor([2.0]).float(), torch.tensor([0.0]).float() )

t0 = torch.autograd.Variable(torch.tensor(0.0).float(), requires_grad = True)

# configure training options
options = {}
options.update({'method': 'Dopri5'})
options.update({'h': None})
options.update({'t0': t0})
options.update({'t1': 1.0})
options.update({'rtol': 1e-3})
options.update({'atol': 1e-5})
options.update({'print_neval': False})
options.update({'neval_max': 1000000})
options.update({'safety': None})
options.update({'t_eval':t_list})

out = odesolve(func, initial_condition, options=options)

plt.plot(out[0][:,0].data.cpu().numpy(),'-o')

plt.show()

loss = torch.sum(out[0] + out[1])
loss.backward()
print('Grad of t0:  {}'.format(t0.grad))
