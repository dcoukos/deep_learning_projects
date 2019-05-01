'''File for implementing module and its descendants'''
import math
from torch import Tensor
import config
import implementation

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def param(self):
        return []

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, act_fn: str) -> None:
        if config.debug:
            print('--- initializing Linear ---')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_fn = act_fn
        self.weights = xavier_initialization(act_fn, in_dim, out_dim)
        self.activation = ReLU()

    def forward(self, *input):
        pass

    def backward(self, *input):
        pass

    def update(self, *input):
        pass


class ReLU(Module):
    pass


def xavier_initialization(act_fn, in_dim: int, out_dim: int, gain=1):
    '''simplified xavier function to initialize weights '''
    if debug:
        print('--- using Xavier ---')
    if act_fn == 'relu':
        gain = math.sqrt(2.0)
    std = gain*math.sqrt(2/(in_dim+out_dim))
    return torch.empty(in_dim, out_dim).normal_(0, std)


class NotImplementedError(Exception):
    print("You forgot to implement forward or backward.")


class DebugError(ValueError):
    print('Debug boolean not defined in implementation.')
