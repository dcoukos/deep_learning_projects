import math
import torch
from torch import Tensor
import config
from optimization import *

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


# TODO: test forward and backward with dummy data.

class Linear(Module):
    '''Write docblock.'''
    def __init__(self, in_dim, out_dim, act_fn):
        if config.show_calls:
            print('--- initializing Linear ---')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights, self.bias = xavier_initialization(act_fn, in_dim,
                                                        out_dim, 'relu')
        self.dl_dw = torch.empty(self.weights.shape)  # Gradients
        self.dl_db = torch.empty(self.bias.shape)
        self.activation = act_fn
        self.prev_x = [] # Prev_x is overwritten for each sample.

    def forward(self, input):  # Why *input vs. input?
        '''Applies forward linear transformation on the data'''
        input = input.view(-1, 1)  # 1D -> 2D tensor for matrix calculations.
        if config.show_calls:
            print('    *Calling Linear.forward()')
        self.prev_x.append(input)
        layer_output = self.weights.mm(input) + self.bias
        act_output = self.activation.forward(layer_output)
        return act_output.squeeze()

    def backward(self, dl_dx):
        # dl_ds represents the effect of the output of this layer on the loss
        # backward pass is intended to be called for each data point.
        if config.show_calls:
            print('    *Calling Linear.backward()')

        dl_ds = self.activation.backward(dl_dx).squeeze()
        # .squeeze should make it work with .mv
        prev_dl_dx = self.weights.t().mv(dl_ds)
        self.dl_dw.add_(dl_ds.view(-1, 1).mm(self.prev_x.view(-1, 1).t()))
        self.dl_db.add_(dl_ds.view(-1, 1))

        if config.show_shapes:
            print('dl_ds shape: ', dl_ds.shape)
            print('weights shape: ', self.weights.shape)
            print('biases shape: ', self.bias.shape)
            print('prev_dl_dx shape: ', self.weights.t().mv(dl_ds).shape)
            print('dl_dw shape: ', (dl_ds.view(-1, 1).mm(
                                    self.prev_x.view(-1, 1).t())).shape)
        return prev_dl_dx.view(-1, 1)

    def update(self, eta, nb_samples):
        '''This function applies one step of the gradient descent, and resets
            the class instance parameters, for the following step.
        '''
        if config.show_calls:
            print('    *Calling Linear.sgd()')

        self.weights = self.weights - (eta*self.dl_dw)/nb_samples
        self.bias = self.bias - (eta*self.dl_db)/nb_samples
        self.dl_dw = torch.empty(self.weights.shape)
        self.dl_db = torch.empty(self.bias.shape)
        self.prev_x = []


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
        return drelu(self.prev_s)*(dl_dx.view(-1, 1))  # TODO: maybe this should be drelu


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
        return dsigma(self.prev_s)*(dl_dx.view(-1, 1))


class Sequential(Module):
    def __init__(self, target, *modules):
        if config.show_calls is None or config.show_shapes is None:
            raise NotImplementedError('You forgot to define debug state.')
        if config.show_calls:
            print('--- Creating a sequential architecture ---')
        self.modules = modules
        self.samples = target.shape[0]
        self.output = torch.empty(self.samples, 10)
        self.target = target
        self.dloss = []

    def forward(self, input):
        if config.show_calls:
            print('    *Calling Sequential.forward()')
        for index, sample in enumerate(input):
            output = sample
            for module in self.modules:
                output = module.forward(output) # TODO: check that this actually updates.

            self.output[index] = output

        return loss(output, self.target).item()

    def backward(self):
        if config.show_calls:
            print('    *Calling Sequential.backward()')
        for index, sample_output in enumerate(self.output):  # dl_dx is the delta loss for each sample
            if config.show_shapes:
                print('   Shape output: ', sample_output.shape)
                print('   Shape target: ', self.target[index].shape)
            dl_dx = dloss(sample_output, self.target[index])
            for module in reversed(self.modules):
                if config.show_shapes:
                    print("dl_dx shape: ", dl_dx.shape)
                dl_dx = module.backward(dl_dx)

    def update(self, eta):
        if config.show_calls:
            print('    *Calling Sequential.update()')
        for module in self.modules:
            module.update(eta, self.samples)
        self.output = torch.empty(self.samples, 10)
        self.dloss = []
