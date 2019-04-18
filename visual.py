import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import dlc_practical_prologue as dlc

train_input, train_target, train_classes, \
test_input, test_target, test_classes = dlc.generate_pair_sets(10)

for imagee in range(0, train_input.size(0)):
    plt.imshow(train_input[2][0].view(14, 14).numpy(), cmap="gray")
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert each image into a torch.FloatTensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the data to have zero mean and 1 stdv
])
train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
plt.imshow(train_set[2][0].view(28, 28).numpy(), cmap="gray")
plt.show()