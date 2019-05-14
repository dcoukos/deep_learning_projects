import config
# from modules import Linear
import torch
from modules import Linear, Sequential, ReLU, Sigma
import dlc_practical_prologue as prologue
from optimization import generate_data

'''
This is the "main" file, and is where the actual architecture is defined.
Additionally, this is where the batch iteration takes place, and where
the learning rates, number of epochs, and other parameters are defined.

'''


# ----- Debugging parameters -----
config.show_calls = False
config.show_shapes = False
torch . set_grad_enabled(False)

# ----- Loading the data -----
train_features, train_label = generate_data(1000)
test_features, test_label = generate_data(1000)


train_features

#  ----- Define the paramters for learning -----
nb_classes = train_label.shape[0]
features = train_features.size(1)
nb_samples = train_features.size(0)
epsilon = 0.000001
eta = 0.1  #nb_samples is now defined in Sequential()
epochs = 100

# Zeta is to make it work correctly with Sigma activation function.
# train_label = train_label.add(0.125).mul(0.8)
# test_label = test_label.add(0.125).mul(0.8)


# ----- Implementation of the architecture -----
architecture = Sequential(Linear(2, 25, ReLU()),
                          Linear(25, 25, ReLU()),
                          Linear(25, 25, ReLU()),
                          Linear(25, 2, Sigma()))

# ----- Training -----
round = 1
for epoch in range(epochs):
    loss, errors = architecture.forward(train_features, train_label)
    architecture.backward()
    architecture.update(eta)
    print(' --- Round ', round, '  Loss: ', loss.item(), '---', ' Errors: ',
          errors, '--- ')
    round += 1
