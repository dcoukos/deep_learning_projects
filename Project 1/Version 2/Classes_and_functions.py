import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as dlc

def split_images(image_pair):
    images1 = image_pair.narrow(1,0,1) # torch.narrow(input_tensor, dimension, start, length) Here: returns the content in image_pair's first dimension indexes from 0 to 1.
    images2 = image_pair.narrow(1,1,1) # Here: returns the content in image_pair's first dimension indexes from 1 to 2.
    return images1, images2

class Net_with_weight_sharing_and_Hot(nn.Module):
# Architecture is inspired more from the model given in the course as example, has close to the 70'000 asked in the project description, and adapted to get pairs of 14x14 images as input.
    def __init__(self):
        super(Net_with_weight_sharing_and_Hot, self).__init__()
        # Image recognition layers (weights are shared (they are used to treat both input images)) :
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)) #14->12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1)) # 12 -> 8
        self.lin1 = nn.Linear(256,200)
        self.lin2 = nn.Linear(200, 10)
        # Digit's value comparison layer :
        self.lin3 = nn.Linear(20,2)

    def forward(self, image_pair): # image_pair is 14x14x2x1
        # Shared-weights part :
        image1, image2 = split_images(image_pair)
        x1 = F.relu(self.conv1(image1)) # 12->12
        x2 = F.relu(self.conv1(image2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=4, stride=4, dilation = 1)) # 12 -> 2
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=4, stride=4, dilation = 1))
        x1 = F.relu(self.lin1(x1.view(-1, 256)))
        x2 = F.relu(self.lin1(x2.view(-1, 256)))
        x1 = self.lin2(x1.view(-1, 200)) # No activation function on the last step
        x2 = self.lin2(x2.view(-1, 200))
        # Output of this : 2 vetors of 10x1

        # Not-shared weights part (Digit's value comparison step) :
        comparison = F.relu(self.lin3(torch.cat((x1, x2), 0).view(-1, 20)))      #  <----   NOT SURE OF THIS !!!

        return comparison, x1, x2 # It must output those (x1, x2 being the digit recognition result) so we can compute the auxiliarry loss (thus rewarding the net if it recognizes well the digits on the image pairs).

def train_model(model, train_image_pairs, train_digit_classes, train_comparison_target, test_image_pairs, test_comparison_target, AUX_LOSS_coef=0.1, batch_size=100, epochs=150, lr = 0.01, printing = True):

    recognition_criterion = torch.nn.CrossEntropyLoss()
    comparison_criterion = torch.nn.CrossEntropyLoss() # We decide here to use the same criterion for the two tasks the networl is doing, but this could as well be changed.
    optimizer = optim.SGD(model.parameters(), lr)

    train_digit_classes1, train_digit_classes2=split_images(train_digit_classes) # Split to compute the auxiliary loss for both digit recognitions.

    for epoch in range(0, epochs):
        sum_loss = 0
        for batch_nb in range(0, train_image_pairs.size(0), batch_size):
            mini_batch = train_image_pairs.narrow(0, batch_nb, batch_size)
            batch_target_x1=train_digit_classes1.narrow(0, batch_nb, batch_size).flatten().long()
            batch_target_x2=train_digit_classes2.narrow(0, batch_nb, batch_size).flatten().long()
            batch_train_comparison_target=train_comparison_target.narrow(0, batch_nb, batch_size).flatten().long()

            comparison, x1, x2 = model(mini_batch) # We evaluate the model on the image pairs in the mini_batch

            # We compute the loss injecting the auxiliary loss, weighted by AUX_LOSS_coef :
            loss = (1-2*AUX_LOSS_coef)* comparison_criterion(comparison, batch_train_comparison_target) + AUX_LOSS_coef*recognition_criterion(x1, batch_target_x1) + AUX_LOSS_coef*recognition_criterion(x2, batch_target_x2)
            sum_loss+=loss.item() # item = to digit
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if printing==True:
            print(('epoch {:d} error rate : {:0.2f}%'.format(epoch, compute_nb_errors(model, test_image_pairs, test_comparison_target, batch_size) / test_image_pairs.size(0) * 100)))

def compute_nb_errors(model, data_input, data_target, mini_batch_size): # This function computes the nb of errors that the model does on the test set
    # I took the function from previous work and didn't adapt it to this net, but this should do.
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1) # WTF
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors
