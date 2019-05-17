import config
# from modules import Linear
import torch
from modules import Linear, Sequential, ReLU, Sigma
import dlc_practical_prologue as prologue
from functions import generate_data
import math
import matplotlib as plt

'''
This is the "main" file, and is where the actual architecture is defined.
Additionally, this is where the batch iteration takes place, and where
the learning rates, number of epochs, and other parameters are defined.

'''


# ----- Debugging parameters -----
config.show_calls = False
config.show_shapes = False
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

# ----- Loading the data -----
train_features, train_labels = generate_data(1000)
test_features, test_labels = generate_data(1000)



#  ----- Define the paramters for learning -----
nb_classes = train_labels.shape[0]
features = train_features.size(1)
nb_samples = train_features.size(0)
epsilon = 0.1
eta = .2  #nb_samples is now defined in Sequential()
batch_size = config.batch_size
epochs = int(config.epochs/(nb_samples/batch_size))


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
prev_loss = math.inf
prev_prev_loss = math.inf
errors = []
for epoch in range(epochs):
    for batch_start in range(0, nb_samples, batch_size):
        features = train_features[batch_start:batch_start+batch_size, :]
        labels = train_labels[batch_start:batch_start+batch_size]
        tr_loss, tr_error = architecture.forward(train_features, train_labels)
        architecture.backward()
        architecture.update(eta)
        loss, error = architecture.forward(test_features, test_labels)
        print(' --- Epoch ', round, '  Test Loss: ', loss.item(), '---', ' Errors: ',
              error)
        if prev_loss < tr_loss:
            eta *= 0.999
            if prev_prev_loss < tr_loss:
                eta *= 0.998

        errors.append(error)
        prev_prev_loss = prev_loss
        prev_loss = tr_loss


        round += 1


loss, error = architecture.forward(test_features, test_labels)
print('#### Test Errors: ', error, ' Test loss: ', loss.item(),
      'Test Accuracy', float(nb_samples-error)*100/nb_samples,'%')
