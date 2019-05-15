import math
import torch
import config
from numpy import heaviside
import modules


def xavier_initialization(in_dim, out_dim, act_fn, gain=1):
    '''simplified xavier function to initialize weights '''
    if config.show_calls:
        print('--- using Xavier ---')
    if isinstance(act_fn, modules.ReLU):
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

# TODO: check that below forms work, track w/ debug


def loss(v, t):
    #print(v.shape, v)
    #print(t.shape, t)
    arg_t = split(t)
    labels = cast_values(arg_t)
    arg_v = split(v.argmax(dim=1))
    assert arg_v.shape == arg_t.shape
    errors = (arg_v.sub_(arg_t)).nonzero().shape[0]/2
    return (v-labels).pow(2).sum(), int(errors)


def split(data):
    output = torch.empty(data.shape[0], 2)
    for ind, value in enumerate(data):
        if value.item() == 1:
            output[ind] = torch.Tensor([0, 1])
        elif value.item() == 0:
            output[ind] = torch.Tensor([1, 0])
        else:
            raise ValueError
    return output


def round_data(data):
    original_shape = data.shape
    output = torch.empty(data.view(-1).shape)
    for ind, value in enumerate(data.view(-1)):
        if value.item() > 0:
            output[ind] = 1
        elif value.item() < 0:
            output[ind] = 0
    output.reshape_(original_shape)
    assert data.view(-1)[0] != output.view(-1)[0]
    return output


def dloss(v, t):
    arg_t = split(t)
    labels = cast_values(arg_t)
    return 2 * (v-labels) # TODO: try t-v, observe behavior.


def generate_data(nb):
    data = torch.Tensor(nb, 2).uniform_(0, 1)
    labels = data.pow(2).sum(1).sub(2/math.pi).mul(-1).sign().add(1).div(2)
    return data, labels


def cast_values(labels):
    shape = labels.shape
    output = torch.empty(labels.view(-1).shape)
    for ind, value in enumerate(labels.view(-1)):
        if value.item() == 1:
            output[ind] = 0.95
        elif value.item() == 0:
            output[ind] = -0.95
        else:
            raise ValueError
    return output.reshape(shape)
