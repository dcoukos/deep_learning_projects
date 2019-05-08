import dlc_practical_prologue as dlc
from shared_arch import *


#Downloading test and train. Splitting the train into train and validation (in order to avoid hyperparameters tuning)

trainVal_input0, trainVal_target0, trainVal_classes0, \
test_input0, test_target0, test_classes0 = dlc.generate_pair_sets(3000) # test_input are the image pairs for testing, test_target are the answers of the comparison of the values of the digit pairs, test_classes are the answers of the digit pairs values
# the 0 stands for pairs (unsplit data)

trainVal_input, _ = split_images(trainVal_input0)
trainVal_classes, _ = split_images(trainVal_classes0)

#Splitting the train into train and validation (in order to avoid hyperparameters tuning)

def split_TrainVal(input):
    N = input.size()[0]
    val = input.narrow(0,0,N//3)
    train = input.narrow(0, N//3, N-N//3)
    return train, val

train_input, val_input = split_TrainVal(trainVal_input)
train_classes, val_classes = split_TrainVal(trainVal_classes)


train_classes0, val_classes0 = split_TrainVal(trainVal_classes0)
train_target0, val_target0 = split_TrainVal(trainVal_target0)


test_input, _ = split_images(test_input0)
test_classes, _ = split_images(test_classes0)

# from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#
# # #training the SharedWeight_Net
# print('Shared Weight Net 2')
# model_shared2 = SharedWeight_Net2()
# model_shared2(train_input.narrow(0, 0, 1))
# print(count_parameters(model_shared2))
# train_model(model_shared2, train_input, train_classes, val_input, val_classes, 100, 50, 0.005)
# #
# #
# # # #training the comparison_Net
# print('Comparison Net Hot')
# net_hot = Comparison_Net_Hot()
# train_model(net_hot, convert_to_hot(train_classes0), train_target0, convert_to_hot(val_classes0), val_target0, 100, 50, 0.2)
# # # # with lr = 0.01 final error 16%, 0.005 31%, 0.05 2.3%, lr = 0.2 0%
#
#
# print('Comparison Net Cold')
# model_comparison = Comparison_Net_Cold()
# train_model(model_comparison, train_classes0.float(), train_target0, val_classes0.float(), val_target0, 100, 50, 0.1)
# # # with lr = 0.01 final error 8.5 %, 0.005 9%, 0.5 0%
# print('test error: {:0.2f}%'.format(compute_nb_errors(model_comparison, test_classes0.float(), test_target0, 50) / test_classes0.size(0) * 100))
#

print('Comparison Net Cold Minimal') # This one freaks out !
net_cold_minimal = Comparison_Net_Cold_Minimal()
train_model(net_cold_minimal, train_classes0.float(), train_target0, val_classes0.float(), val_target0, 100, 150, 0.5)
print('test error: {:0.2f}%'.format(compute_nb_errors(net_cold_minimal, test_classes0.float(), test_target0, 50) / test_classes0.size(0) * 100))

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