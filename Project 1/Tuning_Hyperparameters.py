import dlc_practical_prologue as dlc
from shared_arch import *
from torch.nn import functional as F
import numpy as np # For the linspace function only !

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

# Declaration of the Full net with Weightsharing : 
# Training the Full net with Weightsharing :

def frange(start, stop=None, step=None):
    #Use float number in range() function
    # if stop and step argument is null set start=0.0 and step = 1.0
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0
    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start) # return float number
        start = start + step
        
        
print('Training the Full net with Weightsharing and auxilary Loss= {:0.2f} with varying lr :'.format(AuxilaryLoss))
for lr in np.linspace(lr_min, lr_max, n_lr):
    model=Whole_Shared_Net()
    print('Training with lr= {:0.4f}'.format(lr))
    train_model(model, train_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), batch_size, epochs, lr, printing , True, AuxilaryLoss)

    
def Optimize_parameter(model, lr_range, AuxilaryLoss_range):
    for lr in lr_range:
        for AuxilaryLoss in AuxilaryLoss_range:
            model.reset() # Not sure it exists
            print('Training with lr={:0.2f}, AuxilaryLoss ={:0.2f}'.format(lr, AuxilaryLoss))
            train_model(model, trainVal_images, (train_digit1, train_digit2, train_comparison), val_images, (val_digit1, val_digit2, val_comparison), batch_size, epochs, lr, printing , full, AuxilaryLoss)
            