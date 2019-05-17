import dlc_practical_prologue as dlc
from shared_arch import *
from torch.nn import functional as F
from Custom_Utilities import *



#Downloading test and train. Splitting the train into train and validation (in order to avoid hyperparameters tuning)

trainVal_images, trainVal_comparison, trainVal_digits, \
test_images, test_comparison, test_digits = dlc.generate_pair_sets(3000)

#split train into train and validation
train_images, val_images = split_TrainVal(trainVal_images)
train_digits, val_digits = split_TrainVal(trainVal_digits)
train_comparison, val_comparison = split_TrainVal(trainVal_comparison)

train_image1, train_image2 = split_images(train_images)
train_digit1, train_digit2 = split_images(train_digits)

val_image1, val_image2 = split_images(val_images)
val_digit1, val_digit2 = split_images(val_digits)

test_image1, test_image2 = split_images(test_images)
test_digit1, test_digit2 = split_images(test_digits)



# # #
#training the SharedWeight_Net
#print('Shared Weight Net 2')
#model_shared2 = SharedWeight_Net2()
#print(count_parameters(model_shared2))
# train_model(model_shared2, train_image1, train_digit1, val_image1, val_digit1, 100, 25, 0.005)
#
#
# # #
# # #
# # # # #training the comparison_Net
# print('Comparison Net Hot')
# net_hot = Comparison_Net_Hot()
# train_model(net_hot, convert_to_hot(train_digits), train_comparison, convert_to_hot(val_digits), val_comparison, 100, 25, 0.2)
# # # # # with lr = 0.01 final error 16%, 0.005 31%, 0.05 2.3%, lr = 0.2 0%

print('Comparison Net Full with Shared Weights')
net_full_shared = Whole_Shared_Net()
print(count_parameters(net_full_shared))
train_model(net_full_shared, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), 100, 25, 0.005, printing = True, full = True)

# print('Comparison Net Full without Shared Weights')
# net_full_unshared = Whole_UnShared_Net()
# print(count_parameters(net_full_unshared))
# train_model(net_full_unshared, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), 100, 25, 0.005, full = True)
#
print('Whole FC Net')
net_full_FC = Whole_FC_Net()
print(count_parameters(net_full_FC))
train_model(net_full_FC, train_images, train_comparison, val_images, val_comparison, 100, 25, lr= 0.0005, full = False)

# print('Whole Shared Net Noise Removal')
# net_full_Shared_NoiseFree = Whole_Shared_Net_NoiseRemoval()
# print(count_parameters(net_full_Shared_NoiseFree))
# train_model(net_full_Shared_NoiseFree, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), 100, 25, 0.2, full = True, auxiliaryLoss = 0)
#
print('Comparison without auxiliary loss')
net_full_shared_2 = Whole_Shared_Net()
print(count_parameters(net_full_shared_2))
train_model(net_full_shared_2, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), 100, 25, 0.0005, full = True, auxiliaryLoss = 0)

print('Comparison without auxiliary loss binary')
net_full_binary = Whole_Shared_Net_Binary()
print(count_parameters(net_full_binary))
train_model(net_full_binary, train_images, train_comparison, val_images, val_comparison, 100, 25, 0.0005, full = False, auxiliaryLoss = 0)



#now we try to train first the shared and then the hot with the output of the shared
print('Full Net with Concatenation of already trained parts')

print('Training digit recognition part')
model_shared3 = SharedWeight_Net2()
# print(count_parameters(model_shared2))
train_model(model_shared3, train_image1, train_digit1, val_image1, val_digit1, 100, 25, 0.005)

print('Training comparison on the output with noise elimination')
net_hot_2 = Comparison_Net_Hot()
train_model(net_hot_2, convert_to_hot(torch.cat((torch.argmax(model_shared3(train_image1).detach(), dim = 1, keepdim = True), torch.argmax(model_shared3(train_image2).detach(), dim = 1, keepdim = True)), dim = 1)), train_comparison, convert_to_hot(torch.cat((torch.argmax(model_shared3(val_image1).detach(), dim = 1, keepdim = True), torch.argmax(model_shared3(val_image2).detach(), dim = 1, keepdim = True)), dim = 1)), val_comparison, 100, 25, 0.2)

print('Training comparison on the output without noise elimination')
net_hot_3 = Comparison_Net_Hot()
train_model(net_hot_3, torch.cat((model_shared3(train_image1), model_shared3(train_image2)), dim = 1), train_comparison, torch.cat((model_shared3(val_image1), model_shared3(val_image2)), dim = 1), val_comparison, 100, 25, 0.2)



#
#
# print('Comparison Net Cold')
# model_comparison = Comparison_Net_Cold()
# train_model(model_comparison, train_classes0.float(), train_target0, val_classes0.float(), val_target0, 100, 50, 0.1)
# # # with lr = 0.01 final error 8.5 %, 0.005 9%, 0.5 0%
# print('test error: {:0.2f}%'.format(compute_nb_errors(model_comparison, test_classes0.float(), test_target0, 50) / test_classes0.size(0) * 100))
#

# print('Comparison Net Cold Minimal') # This one freaks out !
# net_cold_minimal = Comparison_Net_Cold_Minimal()
# train_model(net_cold_minimal, train_classes0.float(), train_target0, val_classes0.float(), val_target0, 100, 150, 0.5)
# print('test error: {:0.2f}%'.format(compute_nb_errors(net_cold_minimal, test_classes0.float(), test_target0, 50) / test_classes0.size(0) * 100))

# # # with lr = 0.5 final error 0 %
# #
# # #2do compare results with Net Cold Minimal and with Net Hot (keeping the probabilities for every digit)
# #
# print('Results of the full net (with net cold minimal)')
# Full_Model_Cold=Full_Net_Cold(model_shared2, net_cold_minimal) # Constructing the full net from the above-trained models
# print('error: {:0.2f}%'.format(compute_nb_errors(Full_Model_Cold, test_input0, test_target0, mini_batch_size=100) / test_input0.size(0) * 100))
#
#
# print('Results of the full net (with net cold')
# Full_Model_Cold=Full_Net_Cold(model_shared2, net_cold_minimal) # Constructing the full net from the above-trained models
# print('error: {:0.2f}%'.format(compute_nb_errors(Full_Model_Cold, test_input0, test_target0, mini_batch_size=100) / test_input0.size(0) * 100))
#
#
# #
# print('Results of the full net (with net hot)')
# Full_Model_Hot=Full_Net_Hot(model_shared, net_hot) # Constructing the full net from the above-trained models
# #                                                         # The BUG is here !
# print('error: {:0.2f}%'.format(compute_nb_errors(Full_Model_Hot, test_input0, test_target0, mini_batch_size=100) / test_input0.size(0) * 100))