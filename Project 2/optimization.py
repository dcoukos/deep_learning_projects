import math
import torch
import config
from numpy import heaviside
import modules


def xavier_initialization(in_dim, out_dim, bias, gain=1):
    '''simplified xavier function to initialize weights '''
    if config.show_calls:
        print('--- using Xavier ---')
    if isinstance(bias, modules.ReLU):
        gain = math.sqrt(2.0)
    std = gain*math.sqrt(2/(in_dim+out_dim))
    parameters = torch.empty(out_dim, in_dim + 1).normal_(0, std)
    weights = parameters.narrow(1, 0, in_dim)
    bias = parameters.narrow(1, in_dim, 1)

    if config.show_shapes:
        print('\txavier_initialization shapes\nWeights: ', str(weights.shape),\
              '\nBias: ', str(bias.shape))
    return weights[None, :, :], bias.squeeze()


def sigma(x):
    return x.tanh_()  # redefine using math?


def dsigma(x):
    return 4*(x.exp() + x.mul(-1).exp()).pow(-2)


def drelu(x):
    output = torch.Tensor(x)
    original_shape = output.shape
    output = output.view(-1, 1)
    for index, value in enumerate(output):
        if value.item() <= 0:
            output[index] = 0
        else:
            output[index] = 1
    return output.reshape(original_shape)


def relu(input):
    output = torch.Tensor(input)
    original_shape = output.shape
    output = output.view(-1, 1)
    for index, value in enumerate(output):
        if value.item() < 0:
            output[index] = 0
    return output.reshape(original_shape)


def loss(v, t):
    arg_v = v.argmax(dim=1)
    arg_t = t.argmax(dim=1)
    errors = (arg_v - arg_t).nonzero().shape[0]
    return (v-t).pow(2).sum(), errors


def dloss(v, t):
    return 2 * (v-t) # TODO: try t-v, observe behavior.
