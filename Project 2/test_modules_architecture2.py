# -------- TESTING FORWARD MATH --------------------
from modules import Linear, ReLU, Sequential, Tanh
import torch
from optimization import drelu, dloss

weights = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
weights.shape


input = torch.tensor([[.5, .6],
                      [.1, .2],
                      [.3, .4],
                      [0, .4],
                      [0, .7]])
input.shape

bias = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
bias.shape

'''
input = input[:, :, None]
weights.matmul(input).squeeze() + bias'''

lin = Linear(2, 5, ReLU())

def inside_circle(input):
    import math
    a = input[0]
    b = input[1]
    if math.sqrt(a**2 + b**2) <= 1:
        return True
    else:
        return False

output = torch.tensor([inside_circle(t) for t in input]).long()
output


target = torch.tensor([])
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



# ---- argmax ---

a = torch.Tensor([[1, 2, 2,3 ,2 ,1 , 5, 6,3 ],
                 [1, 2, 2,3 ,2 ,1 , 5, 6,10 ],
                 [1, 2, 2,3 ,2 ,1 , 5, 0,3 ]])

b = torch.Tensor([[1, 12, 2,3 ,2 ,1 , 5, 6,3 ],
                 [1, 2, 2,13 ,2 ,1 , 5, 6,10 ],
                 [1, 2, 2,3 ,2 ,1 , 5, 0,3 ]])
arg_a = a.argmax(dim=1)

arg_b = b.argmax(dim=1)

(arg_a - arg_b).nonzero().shape[0]
