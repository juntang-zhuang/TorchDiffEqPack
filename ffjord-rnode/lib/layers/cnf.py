import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint
from TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve
from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.test_solver_options = {}

    def forward(self, z, logpz=None, reg_states=tuple(), integration_times=None, reverse=False):

        if not len(reg_states)==self.nreg : #and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)


        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # configure training options
        options = {}
        options.update({'method': self.solver})
        options.update({'h': self.solver_options['step_size']})
        options.update({'t0': integration_times[0]})
        options.update({'t1': integration_times[1]})
        options.update({'rtol': [self.rtol, self.rtol] + [1e20] * len(reg_states) })
        options.update({'atol': [self.atol, self.atol] + [1e20] * len(reg_states) })
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        options.update({'t_eval': None})
        options.update({'interpolation_method': 'cubic'})
        options.update({'regenerate_graph': False})
        options.update({'print_time': False})

        if self.training:
            if self.solver in ['sym12async','adalf', 'fixedstep_sym12async','fixedstep_adalf']:
                initial = (z, _logpz) + reg_states
                out = odesolve(self.odefunc, initial, options=options)
                state_t = []
                for _out1, _out2 in zip(initial, out):
                    state_t.append( torch.stack((_out1, _out2),0) )
                state_t = tuple(state_t)
            else:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz) + reg_states,
                    integration_times.to(z),
                    atol=[self.atol, self.atol] + [1e20] * len(reg_states) if self.solver in ['dopri5', 'bosh3'] else self.atol,
                    rtol=[self.rtol, self.rtol] + [1e20] * len(reg_states) if self.solver in ['dopri5', 'bosh3'] else self.rtol,
                    method=self.solver,
                    options=self.solver_options,
                )
        else:
            if self.test_solver in ['sym12async', 'adalf', 'fixedstep_sym12async','fixedstep_adalf']:
                initial = (z, _logpz) + reg_states
                out = odesolve(self.odefunc, initial, options=options)
                state_t = []
                for _out1, _out2 in zip(initial, out):
                    state_t.append(torch.stack((_out1, _out2)))
                state_t = tuple(state_t)
            else:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz),
                    integration_times.to(z),
                    atol=self.test_atol,
                    rtol=self.test_rtol,
                    method=self.test_solver,
                    options=self.test_solver_options,
                )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        reg_states = state_t[2:]

        return z_t, logpz_t, reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
