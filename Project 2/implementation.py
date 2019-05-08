import config
# from modules import Linear
from modules import Linear, Sequential, ReLU, Sigma
import dlc_practical_prologue as prologue

'''
This is the "main" file, and is where the actual architecture is defined.
Additionally, this is where the batch iteration takes place, and where
the learning rates, number of epochs, and other parameters are defined.

'''

# ------------------ TESTING ----------------


# TODO: test forward and backward with dummy data.





# ----- Debugging parameters -----
config.show_calls = False
config.show_shapes = False

# ----- Loading the data -----
train, train_label, test, test_label = prologue.load_data(one_hot_labels=True,
                                                          normalize=True)


#  ----- Define the paramters for learning -----
nb_classes = train_label.size(1)
features = train.size(1)
nb_samples = train.size(0)
zeta = 0.9  # for compatibility with sigma
epsilon = 0.000001
eta = 0.1  #nb_samples is now defined in Sequential()
epochs = 100

# Zeta is to make it work correctly with Sigma activation function.
train_label = train_label*zeta
test_label = test_label*zeta


# ----- Implementation of the architecture -----
architecture = Sequential(Linear(784, 100, ReLU()),
                          Linear(100, 100, ReLU()),
                          Linear(100, 10, Sigma()))


# ----- Training -----
round = 1
for epoch in range(epochs):
    loss = architecture.forward(train, train_label)
    architecture.backward()
    architecture.update(eta)
    print(' --- Round ', round, '  Loss: ', loss, '---')
    round += 1
