import math
import torch
from torch import Tensor
import config
from functions import *

'''File for implementing module and its descendants. These are the classes
    that make up the architectural elements of the network. These classes
    rely on functions which are defined in optimization.py.

    These modules implement forward, backward, and update as instance
    functions.
'''

# TODO, check what's supposed to happen with prev_x in the first layer.
# TODO: check gradients

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
        if config.show_calls:
            print('--- initializing Linear ---')
        self.input = torch.Tensor()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights, self.bias = xavier_initialization(in_dim, out_dim,
                                                        act_fn)
        self.dl_dw = torch.empty(self.weights.shape) # Gradients
        self.dl_db = torch.empty(self.bias.shape)
        self.activation = act_fn
        self.layer_output = torch.Tensor()
        if not isinstance(act_fn, Activation):
            raise RuntimeError('Unacceptable activation function')

    def forward(self, input):  # Why *input vs. input?
        '''Applies forward linear transformation on the data'''
        # Matches the theory.
        input = input[:, :, None]  # 1D -> 2D tensor for matrix calculations.
        self.input = input
        if config.show_calls:
            print('    *Calling Linear.forward()')
        layer_output = self.weights.matmul(input).squeeze() + self.bias
        act_output = self.activation.forward(layer_output)
        return act_output.squeeze()

    def backward(self, dl_dx):
        # dl_ds represents the effect of the output of this layer on the loss
        # backward pass is intended to be called for each data point.
        if config.show_calls:
            print('    *Calling Linear.backward()')
        dl_ds = self.activation.backward(dl_dx)
        prev_dl_dx = self.weights.transpose(1,2).matmul(
                        dl_ds[:, :, None]).squeeze()
        dl_dw = dl_ds[:, :, None].matmul(self.input.transpose(1, 2)).sum(0)
        self.dl_dw = dl_dw/dl_ds.shape[0]  # Normalizes update to the mean
        self.dl_db = dl_ds.sum(0)/dl_ds.shape[0]

        return prev_dl_dx

    def update(self, eta):
        '''This function applies one step of the gradient descent, and resets
            the class instance parameters, for the following step.
        '''
        if config.show_calls:
            print('    *Calling Linear.sgd()')

        self.weights = self.weights - (eta*self.dl_dw)
        self.bias = self.bias - (eta*self.dl_db)
        self.dl_dw = torch.empty(self.weights.shape)
        self.dl_db = torch.empty(self.bias.shape)


# TODO: possibly compress activations by including backward & forward in
#           activation superclass
class ReLU(Module, Activation):
    '''Doc block!'''
    def __init__(self):
        if config.show_calls:
            print('--- initializing ReLU ---')
        self.prev_s = torch.Tensor()

    def forward(self, input):
        if config.show_calls:
            print('    *Calling ReLU.forward()')
        self.prev_s = input
        return relu(input)

    def backward(self, dl_dx):
        '''Sub-derivatives  in {0,1}'''
        if config.show_calls:
            print('   *Calling ReLU.backward()')
        if config.show_shapes:
            print('   shape prev_s: ', self.prev_s.shape)
            print('   shape dl_dx: ', dl_dx.view(-1, 1).shape)  # Nice! 1D ->2D
            # This makes the multiplication work correctly (doesn't make
            # shape 10x10, but instead (10*1)*(10*1) -> (10*1))
        return drelu(self.prev_s)*dl_dx  # TODO: maybe this should be drelu


class Sigma(Module, Activation):
    '''Doc block!'''
    def __init__(self):
        if config.show_calls:
            print('--- initializing Sigma ---')
        self.prev_s = torch.Tensor()

    def forward(self, input):
        if config.show_calls:
            print('    *Calling Sigma.forward()')
        self.prev_s = input
        return sigma(input)

    def backward(self, dl_dx):
        '''Sub-derivatives  in {0,1}'''
        if config.show_calls:
            print('   *Calling Sigma.backward()')
        if config.show_shapes:
            print('   shape prev_s: ', self.prev_s.shape)
            print('   shape dl_dx: ', dl_dx.view(-1, 1).shape)  # Nice! 1D ->2D
            # This makes the multiplication work correctly (doesn't make
            # shape 10x10, but instead (10*1)*(10*1) -> (10*1))
        return dsigma(self.prev_s)*dl_dx


class Sequential(Module):
    def __init__(self, *modules):
        if config.show_calls is None or config.show_shapes is None:
            raise NotImplementedError('You forgot to define debug state.')
        if config.show_calls:
            print('--- Creating a sequential architecture ---')
        self.modules = modules
        self.output = torch.Tensor()
        self.target = torch.Tensor()
        self.dloss = []

    def forward(self, input, target):
        if config.show_calls:
            print('    *Calling Sequential.forward()')
        self.target = target
        output = input
        for module in self.modules:
            output = module.forward(output) # TODO: check that this actually updates.
        self.output = output
        return loss(output, self.target)

    def backward(self):
        if config.show_calls:
            print('    *Calling Sequential.backward()')

        dl_dx = dloss(self.output, self.target)
        for module in reversed(self.modules):
            if config.show_shapes:
                print("dl_dx shape: ", dl_dx.shape)
            dl_dx = module.backward(dl_dx)

    def update(self, eta):
        if config.show_calls:
            print('    *Calling Sequential.update()')
        for module in self.modules:
            module.update(eta)
        self.output = torch.Tensor()
