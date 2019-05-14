import dlc_practical_prologue as dlc
from shared_arch import *
from torch.nn import functional as F
import numpy as np # For the linspace function only !

def Try_parameters(model, lr_range, AuxilaryLoss_range):
    for lr in lr_range:
        for AuxilaryLoss in AuxilaryLoss_range:
            # model.reset() # Not sure it exists, so I do :
            model_temp=model
            print('Training with lr={:0.2f}, AuxilaryLoss ={:0.2f}'.format(lr, AuxilaryLoss))
            train_model(model_temp, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), batch_size, epochs, lr, printing , True, AuxilaryLoss)

            
def Iter_AuxilaryLoss(model, AuxilaryLoss_range): # This function iterates acorss a range of values of the auxiliaryloss (once the lr set), and savec the evolution of the final error rate on digit recognition and comparison of the full net to plot it later. todo: put all the arguments properly
    i=0
    err_digit1=[]
    err_digit2=[]
    err_class=[]
    for AuxilaryLoss in AuxilaryLoss_range:
        i=i+1
        # model.reset() # Not sure it exists, so I do :
        model_temp=model
        print('Training with lr={:0.2f}, AuxilaryLoss ={:0.2f}'.format(lr, AuxilaryLoss))
        train_model(model_temp, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), batch_size, epochs, lr, printing , True, AuxilaryLoss)
        digit1_test_error, digit2_test_error, comparison_test_error = compute_nb_errors(model, test_images, (test_digit1, test_digit2, test_comparison), batch_size, True)
        err_digit1.append(100*digit1_test_error/test_images.size()[0])
        err_digit2.append(100*digit2_test_error/test_images.size()[0])
        err_class.append(100*comparison_test_error/test_images.size()[0])
    return err_digit1, err_digit2, err_class
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

# Training parameters :
batch_size=100
epochs=25
lr_min = 0.001
lr_max= 0.5
n_lr=10
printing =True
AuxilaryLoss=0.1
# Auxiliarryloss :
AuxilaryLoss=0.1
AuxilaryLoss_min=0.05
AuxilaryLoss_max=0.45
n_AuxilaryLoss=10

# Declaration of the model : 
model=Whole_Shared_Net()

# Tuning of the Full net with Weightsharing with Auxilarry Loss: 
print('Tuning for lr. Training the Full net with Weightsharing and auxilarry Loss= {:0.2f} with varying lr :'.format(AuxilaryLoss))
for lr in np.linspace(lr_min, lr_max, n_lr):
    model=Whole_Shared_Net()
    print('Training with lr= {:0.5f}'.format(lr))
    # Format : def train_model(model, train_input, train_target, test_input, test_target, batch_size=100, epochs=150, lr = 0.01, printing = True, full = False, auxiliaryLoss = 0.2)
    train_model(model, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), batch_size, epochs, lr, printing , True, AuxilaryLoss) # So we use the validation set to set the hyperparameters.
    # Cnclusion : lr=0.0060 is the best learning rate, but not by far. It achieves 13.00% error rate on comparison on the test set. With this lr, I try several AuxilaryLoss values : 
    
lr=0.006
print('Tuning for auxiliary loss coefficient. Training the Full net with Weightsharing and lr = {:0.4f} with varying auxilarry Loss :'.format(lr))
# Re-declare :
model=Whole_Shared_Net()
err_digit1, err_digit2, err_class= Iter_AuxilaryLoss(model, np.linspace(AuxilaryLoss_min, AuxilaryLoss_max, n_AuxilaryLoss))
print(err_digit1)
print(err_digit2)
print(err_class)
            
            
# To be continued...