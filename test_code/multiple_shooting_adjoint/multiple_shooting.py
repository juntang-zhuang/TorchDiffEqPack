import torch
import math
import numpy as np
from TorchDiffEqPack import odesolve
from torch import nn

class MultipleShoot(nn.Module):

    def __init__(self, ode_func, chunk_length = 10, observation_length = 100, ODE_options = None,
                 smooth_penalty = 1.0, time_interval = 1.0):
        super(MultipleShoot, self).__init__()
        """
        :param ode_func: The ODE functions, dy/dt = func(t,y)
        :param chunk_length: the observation is divided into chunks, each of of length chunk_length
        :observation_length: total length of observation (This determines how many inter-mediate initial values
         need to specified as extra parameters to update)
        :ODE_options: options for ODE solvers
        :smooth_penalty: penalty for smoothness
        """
        self.odefunc = ode_func
        self.chunk_length = chunk_length
        self.observation_length = observation_length
        self.ODE_options = ODE_options
        self.smooth_penalty = smooth_penalty
        self.time_interval = time_interval

    def prepare_intermediate(self, observations):
        # observations of shape  num_time_points x N, N is the dimension of hidden state y
        observation_length = int( observations.shape[0] )
        self.observation_length = observation_length

        # calculate the number of chunks
        self.num_chunks = math.ceil( float(observation_length) / float(self.chunk_length) )

        # create a list of intermedia results
        self.intermediates = nn.ParameterList()
        for i in range(self.num_chunks):
            self.intermediates.append( nn.Parameter(
                observations[i*self.chunk_length, :], requires_grad=True
                )
            )

    def fit_and_grad(self, observations, time_points): # calculate grad w.r.t parameters
        assert isinstance(time_points, list), "time_points must be of type list"
        # check number of time points match observation
        assert self.observation_length == len(time_points), "Number of time points mismatch observation"

        # create observation into chunks
        data_chunks, time_chunks = [], []
        for i in range(self.num_chunks):
            data_chunks.append(
                observations[ i*self.chunk_length : min( (i+1) * self.chunk_length+1, self.observation_length), :]
            )
            time_chunks.append(
                time_points[ i * self.chunk_length : min( (i+1) * self.chunk_length+1, self.observation_length)]
            )

        # fit data chunk by chunk
        prediction_chunks = []
        for i in range(self.num_chunks):
            data_chunk, time_chunk, intermediate = data_chunks[i], time_chunks[i], self.intermediates[i]

            self.ODE_options.update({'t0': time_chunk[0]})
            self.ODE_options.update({'t1': time_chunk[-1]})
            self.ODE_options.update({'t_eval': time_chunk})

            result = odesolve(self.odefunc, y0 = intermediate, options=self.ODE_options)
            prediction_chunks.append(result)

        return prediction_chunks, data_chunks

    def get_loss(self, prediction_chunks, data_chunks):
        assert len(prediction_chunks)==len(data_chunks), "Length of data_chunks and prediction_chunks must match"

        # loss between prediction and observation
        observation_loss = 0.0
        for data, prediction in zip(data_chunks, prediction_chunks):
            observation_loss = observation_loss + torch.mean((data - prediction)**2)

        # loss in mis-match between prediction and inter-mediate parameters
        mismatch_loss = 0.0
        for i in range(self.num_chunks-1):
            prev, next = prediction_chunks[i][-1,:], self.intermediates[i+1]
            mismatch_loss = mismatch_loss + torch.mean((prev - next)**2)

        loss = observation_loss + mismatch_loss * self.smooth_penalty

        print('Observation loss: {}, smoothness loss {}'.format( observation_loss.item(), mismatch_loss.item() ))

        return loss
