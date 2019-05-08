# -------- TESTING FORWARD MATH --------------------
from modules import Linear, ReLU, Sequential, Sigma
import torch
from optimization import drelu, dloss

weights = torch.Tensor([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]])
weights = weights[None, :, :]
weights.transpose(1,2).shape


input = torch.Tensor([[1, 2, 3, 4, 5],
                      [1, 2, 3, 0, 0],
                      [1, 1, 1, 1, 1]])

bias = torch.Tensor([1, 2, 3, 4])
bias.shape

'''
input = input[:, :, None]
weights.matmul(input).squeeze() + bias'''

lin = Linear(5, 4, ReLU())

output = lin.forward(input)
target = torch.Tensor([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
d_loss = dloss(output, target)

prev_dl_dx = lin.backward(d_loss)

prev_dl_dx.shape

ex_dloss = torch.Tensor([[.1, .2, .2, .1],
                         [.1, .2, .2, .1],
                         [.1, .2, .2, .1]])


dl_ds = drelu(output)*ex_dloss

dl_db = dl_ds.sum()/dl_ds.shape[0]
(drelu(output)*ex_dloss).sum(0)

prev dl_dx = weights.transpose(1,2).matmul(dl_ds[:, :, None]).squeeze()

dl_ds.shape
sample = torch.Tensor([[1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3]])

dl_dw = dl_ds[:, :, None].matmul(sample[:, None, :])
dl_dw.sum(0).shape
