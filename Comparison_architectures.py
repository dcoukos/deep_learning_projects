import dlc_practical_prologue as dlc
from shared_arch import *


# TODO: niel's line

train_input0, train_target0, train_classes0, \
test_input0, test_target0, test_classes0 = dlc.generate_pair_sets(1000)


train_input, _ = split_images(train_input0)
train_classes, _ = split_images(train_classes0)

test_input, _ = split_images(test_input0)
test_classes, _ = split_images(test_classes0)

#
#

# #training the SharedWeight_Net
print('Shared Weight Net')
model = SharedWeight_Net()
train_model(model, train_input, train_classes, test_input, test_classes, 100, 150, 0.01)

#training the comparison_Net
print('Comparison Net Hot')
model_comparison = Comparison_Net_Hot()
train_model(model_comparison, convert_to_hot(train_classes0), train_target0, convert_to_hot(test_classes0), test_target0, 100, 150, 0.2)
# with lr = 0.01 final error 16%, 0.005 31%, 0.05 2.3%, lr = 0.2 0%

print('Comparison Net Cold')
model_comparison = Comparison_Net_Cold()
train_model(model_comparison, train_classes0.float(), train_target0, test_classes0.float(), test_target0, 100, 150, 0.05)
# with lr = 0.01 final error 8.5 %, 0.005 9%, 0.5 0%

print('Comparison Net Cold Minimal')
model_comparison = Comparison_Net_Cold_Minimal()
train_model(model_comparison, train_classes0.float(), train_target0, test_classes0.float(), test_target0, 100, 150, 0.5)
# with lr = 0.5 final error 0 %

#2do compare results with Net Cold Minimal and with Net Hot (keeping the probabilities for every digit)