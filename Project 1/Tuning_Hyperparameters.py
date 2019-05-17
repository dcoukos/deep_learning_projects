import dlc_practical_prologue as dlc
from shared_arch import *
from torch.nn import functional as F



#  import numpy as np # For the linspace function only !

def Scan_parameters(model_class, lr_range, AuxilaryLoss_range, printing, full, val_images, val_digit1, val_digit2,
                    val_comparison, epochs):  # horrible function to scan the lr and auxilary loss parameters and return a tensor with the results.

    val_digit1_error = torch.zeros(lr_range.size(0), AuxilaryLoss_range.size(0))
    val_digit2_error = torch.zeros(lr_range.size(0), AuxilaryLoss_range.size(0))
    val_class_error = torch.zeros(lr_range.size(0), AuxilaryLoss_range.size(0))

    i = 0
    for lr in lr_range:
        j = 0
        for AuxilaryLoss in AuxilaryLoss_range:
            print('Training with lr={:0.4f}, AuxilaryLoss ={:0.4f}'.format(lr, AuxilaryLoss))
            # model.reset() # Not sure it exists, so I do :
            model = model_class()

            if full:
                train_model(model, train_images, (train_digit1, train_digit2, train_comparison), val_images,
                            (val_digit1, val_digit2, val_comparison), batch_size, epochs, lr, printing, True,
                            AuxilaryLoss)
                digit1_val_error, digit2_val_error, comparison_val_error = compute_nb_errors(model, val_images, (
                val_digit1, val_digit2, val_comparison), batch_size, True)

                val_digit1_error[i][j] = (100 * digit1_val_error / val_images.size()[0])
                val_digit2_error[i][j] = (100 * digit2_val_error / val_images.size()[0])
                val_class_error[i][j] = (100 * comparison_val_error / val_images.size()[0])

            if not full:
                #model, train_input, train_target, test_input, test_target, batch_size=100, epochs=150, lr = 0.01, printing = True, full = False, auxiliaryLoss = 0.2

                train_model(model, train_images, train_comparison, val_images, val_comparison, batch_size, epochs, lr, printing = False, full = False)

                comparison_val_error = compute_nb_errors(model, val_images, val_comparison, batch_size, False)
                val_class_error[i][j] = (100 * comparison_val_error / val_images.size()[0])

            j = j + 1

        i = i + 1

    if full:
        return val_digit1_error, val_digit2_error, val_class_error

    elif not full:
        return val_class_error



# Training parameters :

runs = 3
epochs = 25 #Pao
lr_min = 0.0001 #Pao proposes 0.01
lr_max = 0.1
n_lr = 10
lr_range = torch.logspace(torch.log10(torch.tensor(lr_min)), torch.log10(torch.tensor(lr_max)), n_lr)
printing = True
#  Auxiliarryloss :
AuxilaryLoss_min = 0
AuxilaryLoss_max = 0
n_AuxilaryLoss = 1
AuxilaryLoss_range = torch.linspace(AuxilaryLoss_min, AuxilaryLoss_max, n_AuxilaryLoss)


err_class = torch.zeros(runs, n_lr, n_AuxilaryLoss)
err_digit1 = torch.zeros(runs, n_lr, n_AuxilaryLoss)
err_digit2 = torch.zeros(runs, n_lr, n_AuxilaryLoss)

for i in range(runs):
    # downloading the data

    # Downloading test and train. Splitting the train into train and validation (in order to avoid hyperparameters tuning)

    trainVal_images, trainVal_comparison, trainVal_digits, \
    test_images, test_comparison, test_digits = dlc.generate_pair_sets(3000)

    # split train into train and validation
    train_images, val_images = split_TrainVal(trainVal_images)
    train_digits, val_digits = split_TrainVal(trainVal_digits)
    train_comparison, val_comparison = split_TrainVal(trainVal_comparison)

    train_image1, train_image2 = split_images(train_images)
    train_digit1, train_digit2 = split_images(train_digits)

    val_image1, val_image2 = split_images(val_images)
    val_digit1, val_digit2 = split_images(val_digits)

    test_image1, test_image2 = split_images(test_images)
    test_digit1, test_digit2 = split_images(test_digits)

    batch_size = 100





    #err_digit1[i], err_digit2[i], err_class[i] = Scan_parameters(Whole_Shared_Net, lr_range, AuxilaryLoss_range, False, True, val_images, val_digit1, val_digit2, val_comparison, epochs)

err_class[i] = Scan_parameters(Whole_FC_Net, lr_range, AuxilaryLoss_range, False, False, val_images, val_digit1, val_digit2, val_comparison, epochs)


#
# with open("tuning_hyperparameters_Shared_Net.txt", "a") as f:
#     print('error on digit 1 : ', file=f)
#     print(torch.mean(err_digit1, 0), file=f)
#     print(torch.std(err_digit1, 0), file=f)
#     print('error on digit 2 : ', file=f)
#     print(torch.mean(err_digit2, 0), file=f)
#     print(torch.std(err_digit2, 0), file=f)
#     print('error on their comparison : ', file=f)
#     print(torch.mean(err_class , 0), file=f)
#     print(torch.std(err_class, 0), file=f)
#     print('with auxiliary loss : ', file=f)
#     print(AuxilaryLoss_range, file=f)
#     print( 'with lr : ', file=f)
#     print(lr_range, file=f)

with open("tuning_hyperparameters_FC.txt", "a") as f:

    print('error on their comparison : ', file=f)
    print(torch.mean(err_class , 0), file=f)
    print(torch.std(err_class, 0), file=f)
    print('with auxiliary loss : ', file=f)
    print(AuxilaryLoss_range, file=f)
    print( 'with lr : ', file=f)
    print(lr_range, file=f)



# print('Whole FC Net')
# lr_min = 0.0005
# lr_max = 0.5
# n_lr = 2
# lr_range = torch.logspace(torch.log10(torch.tensor(lr_min)), torch.log10(torch.tensor(lr_max)), n_lr)
#
# #dummy
# AuxilaryLoss_min = 0
# AuxilaryLoss_max = 0
# n_AuxilaryLoss = 1
# AuxilaryLoss_range = torch.linspace(AuxilaryLoss_min, AuxilaryLoss_max, n_AuxilaryLoss)
#
#
# err_class = Scan_parameters(Whole_FC_Net, lr_range, AuxilaryLoss_range, False, False, val_images,
#                                                     val_digit1, val_digit2, val_comparison, epochs= 25)
# print('learning-rates range:')
# print(lr_range)
# print('Error class:')
# print(err_class)
# print('Best learning-rate for comparison:')
# print(lr_range[torch.argmin(err_class)])

#f.close()

