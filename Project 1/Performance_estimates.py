import torch
from shared_arch import *
import dlc_practical_prologue as dlc


def performance_estimator(model_class, lr, AuxilaryLoss, full = False):

    batch_size = 100
    epochs = 25 ##2 modify for testing
    rounds = 10

    list_digit1_train_error = torch.zeros(rounds)
    list_digit2_train_error = torch.zeros(rounds)
    list_comparison_train_error = torch.zeros(rounds)
    list_digit1_test_error = torch.zeros(rounds)
    list_digit2_test_error = torch.zeros(rounds)
    list_comparison_test_error = torch.zeros(rounds)

    for i in range(rounds):

        #generating randomized train and test sets
        train_images, train_comparison, train_digits, \
        test_images, test_comparison, test_digits = dlc.generate_pair_sets(1000)

        train_digit1, train_digit2 = split_images(train_digits)
        test_digit1, test_digit2 = split_images(test_digits)

        size_train = train_images.size()[0]
        size_test = test_images.size()[0]

        #instantiate new model
        model = model_class()

        if full:

            train_model(model, train_images, (train_digit1, train_digit2, train_comparison), test_images,
                        (test_digit1, test_digit2, test_comparison), batch_size, epochs, lr, False, True,
                        AuxilaryLoss)

            list_digit1_train_error[i], list_digit2_train_error[i], list_comparison_train_error[i] = compute_nb_errors(model, train_images, (
                train_digit1, train_digit2, train_comparison), batch_size, True)

            list_digit1_test_error[i], list_digit2_test_error[i], list_comparison_test_error[i] = compute_nb_errors(model, test_images, (
                test_digit1, test_digit2, test_comparison), batch_size, True)
        else:

            train_model(model, train_images, train_comparison, test_images,
                        test_comparison, batch_size, epochs, lr, False, False,
                        AuxilaryLoss)

            list_comparison_train_error[i] = compute_nb_errors(model, train_images, train_comparison, batch_size, False)

            list_comparison_test_error[i] = compute_nb_errors(model, test_images, test_comparison, batch_size, False)

    #converting number of errors into percentage
    list_comparison_train_error =  100*list_comparison_train_error/ size_train
    list_digit1_train_error = 100*list_digit1_train_error / size_train
    list_digit2_train_error = 100*list_digit2_train_error / size_train

    list_comparison_test_error =  100*list_comparison_test_error/ size_test
    list_digit1_test_error = 100*list_digit1_test_error / size_test
    list_digit2_test_error = 100*list_digit2_test_error / size_test

    #returning mean and the standard deviation
    if full:
        print('train accuracy [%]: digit1 {:0.2f} \u00B1 {:0.2f}, digit2 {:0.2f}% \u00B1 {:0.2f}, comparison {:0.2f} \u00B1 {:0.2f}'.format(torch.mean(list_digit1_train_error), torch.std(list_digit1_train_error), torch.mean(list_digit2_train_error), torch.std(list_digit2_train_error), torch.mean(list_comparison_train_error)
                                                                                           , torch.std(list_comparison_train_error)))
        print('test accuracy [%]: digit1 {:0.2f} \u00B1 {:0.2f}, digit2 {:0.2f}% \u00B1 {:0.2f}, comparison {:0.2f} \u00B1 {:0.2f}'.format(torch.mean(list_digit1_test_error), torch.std(list_digit1_test_error), torch.mean(list_digit2_test_error), torch.std(list_digit2_test_error), torch.mean(list_comparison_test_error), torch.std(list_comparison_test_error)))

    else:
        print(
            'train accuracy [%]: comparison {:0.2f} \u00B1 {:0.2f}'.format(
                torch.mean(list_comparison_train_error)
                , torch.std(list_comparison_train_error)))
        print(
            'test accuracy [%]: comparison {:0.2f} \u00B1 {:0.2f}'.format(
                torch.mean(list_comparison_test_error), torch.std(list_comparison_test_error)))


def separate_net_performance_estimator(lr, NoiseFree):
    #not very elegant, but our previous function did not adapt well to sepately trainable architectures

    batch_size = 100
    epochs = 1 ##2 modify for testing
    rounds = 2

    list_digit1_train_error = torch.zeros(rounds)
    list_digit2_train_error = torch.zeros(rounds)
    list_comparison_train_error = torch.zeros(rounds)
    list_digit1_test_error = torch.zeros(rounds)
    list_digit2_test_error = torch.zeros(rounds)
    list_comparison_test_error = torch.zeros(rounds)

    for i in range(rounds):

        #generating randomized train and test sets
        train_images, train_comparison, train_digits, \
        test_images, test_comparison, test_digits = dlc.generate_pair_sets(1000)

        train_digit1, train_digit2 = split_images(train_digits)
        test_digit1, test_digit2 = split_images(test_digits)
        train_image1, train_image2 = split_images(train_images)
        test_image1, test_image2 = split_images(test_images)

        size_train = train_images.size()[0]
        size_test = test_images.size()[0]

        #instantiate new model
        net_digit_recognition = SharedWeight_Net2()
        net_comparison = Comparison_Net_Hot()

        #training first part
        train_model(net_digit_recognition, train_image1, train_digit1, test_image1, test_digit1, 100, epochs, lr)

        #training second part

        if NoiseFree:
            train_intermediate = convert_to_hot(torch.cat((torch.argmax(net_digit_recognition(train_image1).detach(), dim=1,
                                                                      keepdim=True),
                                                         torch.argmax(net_digit_recognition(train_image2).detach(), dim=1,
                                                                      keepdim=True)), dim=1))
            test_intermediate = convert_to_hot(
                torch.cat((torch.argmax(net_digit_recognition(test_image1).detach(), dim=1,
                                        keepdim=True),
                           torch.argmax(net_digit_recognition(test_image2).detach(), dim=1,
                                        keepdim=True)), dim=1))
        else:
            train_intermediate = torch.cat((net_digit_recognition(train_image1).detach(), net_digit_recognition(train_image2).detach()), dim = 1)
            test_intermediate = torch.cat((net_digit_recognition(test_image1).detach(), net_digit_recognition(test_image2).detach()), dim = 1)

        train_model(net_comparison, train_intermediate, train_comparison,
                    test_intermediate, test_comparison, 100, 25, 0.2)


        list_digit1_train_error[i] = compute_nb_errors(net_digit_recognition, train_image1, train_digit1, batch_size, False)
        list_comparison_train_error[i] = compute_nb_errors(net_comparison, train_intermediate, train_comparison, 100)

        list_digit1_test_error[i] = compute_nb_errors(net_digit_recognition, test_image1, test_digit1, batch_size, False)
        list_comparison_test_error[i] = compute_nb_errors(net_comparison, test_intermediate, test_comparison, 100)

    #converting number of errors into percentage
    list_comparison_train_error =  list_comparison_train_error/ size_train
    list_digit1_train_error = list_digit1_train_error / size_train

    list_comparison_test_error =  list_comparison_test_error/ size_test
    list_digit1_test_error = list_digit1_test_error / size_test

    #returning mean and the standard deviation

    print('train accuracy [%]: digit1 {:0.2f} \u00B1 {:0.2f}, comparison {:0.2f} \u00B1 {:0.2f}'.format(torch.mean(list_digit1_train_error), torch.std(list_digit1_train_error), torch.mean(list_comparison_train_error)
                                                                                           , torch.std(list_comparison_train_error)))
    print('test accuracy [%]: digit1 {:0.2f} \u00B1 {:0.2f}, comparison {:0.2f} \u00B1 {:0.2f}'.format(torch.mean(list_digit1_test_error), torch.std(list_digit1_test_error), torch.mean(list_comparison_test_error), torch.std(list_comparison_test_error)))





print('Performance Estimates')

print('\nWhole Shared Net')
lr=0.004
auxl=0.20
performance_estimator(Whole_Shared_Net, lr, auxl, full = True) #2DO: choose best learning rate and auxiliary loss # Niels: this was done previously (bar graph) and it gives lr=0.005 and Auxiliary_loss=0.17 optimally 

print('\nWhole FC Net')
performance_estimator(Whole_FC_Net, 0.0005, 0.2, full = False) #2DO: choose best learning rate

print('\nComparison Net Full without Shared Weights')
performance_estimator(Whole_UnShared_Net, 0.0005, 0.2, full = True) #2DO: choose best learning rate (use same auxiliary loss than whole shared net or i don't know)

print('\nWhole Shared Net Noise Removal')
performance_estimator(Whole_Shared_Net_NoiseRemoval, 0.0005, 0.2, full = True) #same

print('\nFull Net with Concatenation of already trained parts Noise Free')
separate_net_performance_estimator(lr= 0.005, NoiseFree= True)

print('\nFull Net with Concatenation of already trained parts')
separate_net_performance_estimator(lr= 0.005, NoiseFree= False)