'''File for implementing module and its descendants'''
import math
import torch
from torch import Tensor
import config
from optimization import *


# TODO: get .pyi files for pytorch and static check with type hints.


class Activation:
    pass


class Module(object):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    '''Write docblock.'''
    def __init__(self, in_dim, out_dim, act_fn):
        if config.debug:
            print('--- initializing Linear ---')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights, self.bias = xavier_initialization(act_fn, in_dim,
                                                        out_dim, 'relu')
        self.prev_x = torch.Tensor()
        self.dl_dw = torch.empty(self.weights.shape)
        self.dl_db = torch.empty(self.bias.shape)

    def forward(self, input):  # Why *input vs. input?
        '''Applies forward linear transformation on the data'''
        if config.debug:
            print('\t *Calling Linear.forward()')
        self.prev_x = input
        # DEBUG
        if config.debug:
            print('weights shape: ', str(self.weights.shape))
            print('input shape: ', str(input.shape))
            print('output shape before bias: ', str(self.weights.mv(input).shape))
            print('bias shape: ', str(self.bias.shape))
            print('output shape: ', str((self.weights.mv(input) + self.bias.shape)))
        return self.weights.mv(input) + self.bias

    def backward(self, dl_ds):
        #dl_ds represents the effect of the output of this layer on the loss
        # backward pass is intended to be called for each data point.
        if config.debug:
            print('\t *Calling Linear.backward()')
        prev_dl_dx = self.weights.t().mv(dl_ds)
        self.dl_dw.add_(dl_ds.view(-1, 1).mm(prev_x.view(-1, 1)))
        self.dl_db.add_(dl_ds)
        return prev_dl_dx

    def sgd(eta):
        if config.debug:
            print('\t *Calling Linear.sgd()')
        self.weights = self.weights - eta*self.dl_dw
        self.bias = self.bias - eta*self.dl_db

    def update(self, *input):
        pass


class ReLU(Module, Activation):
    '''Doc block!'''
    def __init__(self):
        if config.debug:
            print('--- initializing ReLU ---')
        self.prev_s = torch.Tensor()

    def forward(self, input):
        if config.debug:
            print('\t *Calling ReLU.forward()')
        self.prev_s = input
        return relu(input)

    def backward(self, dl_dx):
        '''Sub-derivatives  in {0,1}'''
        if config.debug:
            print('\t *Calling ReLU.backward()')
        return relu(self.prev_s)*(dl_dx)


class Sequential(Module):
    def __init__(self, target, *modules):
        if config.debug is None:
            raise NotImplementedError('You forgot to define debug state.')
        if config.debug:
            print('--- Creating a sequential architecture ---')
        self.modules = modules
        if not isinstance(self.modules[-1], Activation):
            raise Exception('Last module must be an activation function')
        self.output = torch.Tensor()
        self.target = target

    def forward(self, input):
        if config.debug:
            print('\t *Calling Sequential.forward()')
        for sample in input:
            output = sample
            for module in self.modules:
                output = module.forward(output)
        self.output = output
        return loss(output, self.target)  # TODO: confirm this value gets reassigned.

    def backward(self):
        if config.debug:
            print('\t *Calling Sequential.backward()')
        dl_dx = dloss(self.output, self.target)
        for module in reversed(self.modules):
            module.backward(dl_dx)

    def sgd(self, eta):
        if config.debug:
            print('\t *Calling Sequential.sgd()')
        for module in self.modules:
            module.sgd(eta)
