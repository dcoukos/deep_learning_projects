from Classes_and_functions import *
import dlc_practical_prologue as dlc
# from torchsummary import summary

# Generate the train/balidation/test data :
train_image_pairs, train_comparison_target, train_digit_classes,\
test_image_pairs , test_comparison_target, _= dlc.generate_pair_sets(1000)

# Declaration of the net : 
model=Net_with_weight_sharing_and_Hot()

# Training parameters :
batch_size=100
epochs=50
lr = 1
printing =True
AUX_LOSS_coef=0.1
# Display :
# summary(model, (2, 14, 14))
# Training :
print('Training full net')
train_model(model, train_image_pairs, train_digit_classes, train_comparison_target, test_image_pairs, test_comparison_target, AUX_LOSS_coef, batch_size, epochs, lr, printing)