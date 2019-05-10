from Classes_and_functions import *
import dlc_practical_prologue as dlc
import shared_arch
# from torchsummary import summary

# Generate the train/balidation/test data :
train_image_pairs, train_comparison_target, train_digit_classes,\
test_image_pairs , test_comparison_target, _= dlc.generate_pair_sets(3000)

# Declaration of the net : 
model = Net_with_weight_sharing_and_Hot()

# Training parameters :
batch_size=10
epochs=50
lr = 0.005
printing = True
AUX_LOSS_coef = 0.5

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Shared Weight Net 2')
model_shared2 = shared_arch.SharedWeight_Net2()
print(count_parameters(model_shared2))
shared_arch.train_model(model_shared2, train_image_pairs, train_digit_classes, test_image_pairs, test_comparison_target, 100, 50, 0.005)


#let us test the compute_nb_errors
comparison_hot, digits1_hot, digits2_hot = model(train_image_pairs)
_, train_comparison_fake_target = torch.max(comparison_hot, 1)
_, digits1 = torch.max(digits1_hot, 1)
_, digits2 = torch.max(digits2_hot, 1)
tot_samples = train_image_pairs.size(0)
nb_comparison_errors, nb_digitRecognition1_errors, nb_digitRecognition2_errors = compute_nb_errors(model,
                                                                                                   train_image_pairs,
                                                                                                   train_comparison_fake_target,
                                                                                                   digits1,
                                                                                                   digits2,
                                                                                                   batch_size)  #  This function computes the nb of errors that the model does on the test set

print(('epoch {:d} train error: comparison: {:0.2f}% digit1: {:0.2f}% digit2: {:0.2f}%'.format(0,  nb_comparison_errors/tot_samples * 100, nb_digitRecognition1_errors/tot_samples * 100, nb_digitRecognition2_errors/tot_samples * 100)))

# Display :
# summary(model, (2, 14, 14))
# Training :
print('Training full net')
Classes_and_functions.train_model(model, train_image_pairs, train_digit_classes, train_comparison_target, test_image_pairs, test_comparison_target, AUX_LOSS_coef, batch_size, epochs, lr, printing)