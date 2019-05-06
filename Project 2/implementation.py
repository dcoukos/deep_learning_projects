import config
# from modules import Linear
from modules import Linear, Sequential, ReLU
import dlc_practical_prologue as prologue


config.show_calls = False
config.show_shapes = False

train, train_label, test, test_label = prologue.load_data(one_hot_labels=True,
                                                          normalize=True)

nb_classes = train_label.size(1)
features = train.size(1)
nb_samples = train.size(0)
zeta = 0.9  # for compatibility with sigma
epsilon = 0.000001
eta = 0.1/nb_samples
epochs = 100


train_label = train_label*zeta
test_label = test_label*zeta
# TODO: Implement multilayered architecture
architecture = Sequential(train_label,
                          Linear(784, 100, 'relu'),
                          Linear(100, 100, 'relu'),
                          Linear(100, 10, 'sigma'))


# TODO: update over 100 epochs.
round = 1
for epoch in range(3):
    loss = architecture.forward(train)
    architecture.backward()
    architecture.update(eta)
    print(' --- Round ', round, '  Loss: ', loss, '---')
    round += 1
