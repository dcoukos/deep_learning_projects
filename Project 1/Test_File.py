import dlc_practical_prologue as dlc
from shared_arch import *

# Generate the train/balidation/test data :
train_input, train_target, train_classes, \
test_input, test_target, test_classes = dlc.generate_pair_sets(1000)
    # I find it too ugly the way it's done to generate the validation dataset, therefore i don't use this until it will be incorporated in dlc.generate_pair_sets, or in a new version of this function (could be called dlc.generate_pair_sets_validation)
train_images1, train_images2=split_images(train_input) # 1 is for the first igit to recognize, 2 is for the second.
train_classes1, train_classes2=split_images(train_classes)
test_images1, test_images2 = split_images(test_input)
test_classes1, test_classes2 = split_images(test_classes)

# Declaration of the nets :
digit_recognition_net=SharedWeight_Net2() # We choose this class, because it is the one among those we tried giving the best perf.
comparison_net=Comparison_Net_Hot() # We choose this Hot, because it is as well the best performer.
# Will come after training : Total_net=Full_Net_Hot(digit_recognition_net, comparison_net)

# Training parameters :
batch_size=100
epochs=150
lr = 0.01
printing =True

# Training :
print('Training the digit recognition part of the full net')
train_model(digit_recognition_net, train_images1, train_classes1, test_images1, test_classes1, batch_size, epochs, lr, printing)
print('Test error: {:0.2f}%'.format(compute_nb_errors(net_cold_minimal, test_classes0.float(), test_target0, 50) / test_classes0.size(0) * 100)) # For display.


    