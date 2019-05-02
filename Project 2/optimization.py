import math
import torch
import config


def xavier_initialization(act_fn, in_dim, out_dim, bias, gain=1):
    '''simplified xavier function to initialize weights '''
    if config.debug:
        print('--- using Xavier ---')
    if act_fn == 'relu':
        gain = math.sqrt(2.0)
    std = gain*math.sqrt(2/(in_dim+out_dim))
    parameters = torch.empty(out_dim, in_dim + 1).normal_(0, std)
    weights = parameters.narrow(1, 0, in_dim)
    bias = parameters.narrow(1, in_dim, 1)
    if config.debug:
        print('\txavier_initialization shapes\nWeights: ', str(weights.shape),\
              '\nBias: ', str(bias.shape))
    return weights, bias


def sigma(x):
    return x.tanh_()


def dsigma(x):
    return 4*(x.exp() + x.mul(-1).exp()).pow(-2)


def relu(input):
    output = torch.Tensor(input)
    original_shape = output.shape
    output = output.view(-1, 1)
    for value in output:
        if value.item() < 0:
            value = 0
    return output.view(original_shape)


def loss(v, t):
    return (v-t).pow(2).sum()


def dloss(v, t):
    return 2 * (v - t)
