import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import operator




# Code for narrowing to first image
def split_images(data):
    images1 = data.narrow(1,0,1)
    images2 = data.narrow(1,1,1)
    return images1, images2

class SharedWeight_Net(nn.Module):
    #takes as input a 14x14 image and returns a tensor with 10 entries for 10 class scores
    def __init__(self):
        super(SharedWeight_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1)) #14->12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)) # 6-> 4
        self.lin1 = nn.Linear(256,120)
        self. lin2 = nn.Linear(120,84)
        self.lin3 = nn.Linear(84,10)
        # self.out = nn.Linear(20, 1) #TODO: test w/ addtional Output Layer

    def forward(self, x):
        x = F.relu(self.conv1(x)) #12->12
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2, dilation = 1)) #4 -> 2
        x = F.relu(self.lin1(x.view(-1, 256)))
        x = F.relu(self.lin2(x.view(-1, 120)))
        x = F.relu(self.lin3(x.view(-1, 84)))
        return x

class SharedWeight_Net2(nn.Module): # This is different from SharedWeight_Net, it is inspired more from the model given in the course as example, and has more parameters ( closer to the 70'000 asked in the project description).
    def __init__(self):
        super(SharedWeight_Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)) #14->12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1)) # 12 -> 8
        self.lin1 = nn.Linear(256,200)
        self.lin2 = nn.Linear(200, 10)
        # self.out = nn.Linear(20, 1) #TODO: test w/ addtional Output Layer

    def forward(self, x):
        x = F.relu(self.conv1(x)) #12->12
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=4, stride=4, dilation = 1)) # 12 -> 2
        x = F.relu(self.lin1(x.view(-1, 256)))
        x = self.lin2(x.view(-1, 200))
        return x

class Comparison_Net_Hot(nn.Module):
    # this module takes as input a hot vector of size 20 (with all zeros and 1 at the place of the correct class) from the shared_weight net
    # and returns two activations (that will correspond to "bigger" neuron or to "smaller or/equal" neuron)
    def __init__(self):
        super(Comparison_Net_Hot, self).__init__()
        self.lin = nn.Linear(20,2)

    def forward(self, x):
        x = F.relu(self.lin(x.view(-1, 20)))
        return x

class Whole_Shared_Net(nn.Module):
    def __init__(self):
        super(Whole_Shared_Net, self).__init__()
        self.sharedNet = SharedWeight_Net2()
        self.comparisonNet = Comparison_Net_Hot()
    def forward(self, x):
        images1, images2 = split_images(x)
        digit1_hot = self.sharedNet(images1)
        digit2_hot = self.sharedNet(images2)
        #before x = self.comparisonNet(torch.cat((digit1_hot, digit2_hot), dim=1))
        x = self.comparisonNet(torch.cat((digit1_hot, digit2_hot), dim=1))
        return digit1_hot, digit2_hot, x

class Whole_UnShared_Net(nn.Module):
    def __init__(self):
        super(Whole_UnShared_Net, self).__init__()
        self.sharedNet1 = SharedWeight_Net2()
        self.sharedNet2 = SharedWeight_Net2()
        self.comparisonNet = Comparison_Net_Hot()
    def forward(self, x):
        images1, images2 = split_images(x)
        digit1_hot = self.sharedNet1(images1)
        digit2_hot = self.sharedNet2(images2)
        #before x = self.comparisonNet(torch.cat((digit1_hot, digit2_hot), dim=1))
        x = self.comparisonNet(torch.cat((digit1_hot, digit2_hot), dim=1))
        return digit1_hot, digit2_hot, x


class Comparison_Net_Cold(nn.Module):
    # this module takes as input a hot vector of size 2 (with the index of the correct class)
    # -> can be replaced by Comparison_Net_Cold_Minimal
    def __init__(self):
        super(Comparison_Net_Cold, self).__init__()
        self.lin1 = nn.Linear(2,10)
        self.lin2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x.view(-1, 2)))
        x = F.relu(self.lin2(x.view(-1, 10)))
        return x

class Comparison_Net_Cold_Minimal(nn.Module):
    # this module takes as input a hot vector of size 2 (with the index of the correct class)
    def __init__(self):
        super(Comparison_Net_Cold_Minimal, self).__init__()
        self.lin1 = nn.Linear(2,6)
        self.lin2 = nn.Linear(6,2)

    def forward(self, x):
        x = F.relu(self.lin1(x.view(-1, 2)))
        x = F.relu(self.lin2(x.view(-1, 6)))
        return x

class Full_Net_Cold(nn.Module): # Niels: this is the combinated full net, that takes digit images (as couples) as input and outputs the comparison result. It is crafted from the two already trained individual networks that we already did: digit_classification_model and comparison_model, taken as inputs in the constructor
    def __init__(self, digit_classification_model, comparison_model):
        super(Full_Net_Cold, self).__init__() # I don't know the purpose of this line but i put it
        self.digit_classification_model=digit_classification_model
        self.comparison_model=comparison_model
    def forward(self, x):
        image1, image2 =split_images(x)
        x1=self.digit_classification_model(image1) #returns a tensor of size n x 10
        x2=self.digit_classification_model(image2) #returns a tensor of size n x 10
        # we want to get n x 2 with the index
        x =  convert_hothot_to_digitdigit(x1, x2)
        return self.comparison_model(x) # Might need to put a torch.cat((x1, x2), dim=0) here to feed the comparison model...

class Full_Net_Hot(nn.Module): # Niels: this is the combinated full net, that takes digit images (as couples) as input and outputs the comparison result. It is crafted from the two already trained individual networks that we already did: digit_classification_model and comparison_model, taken as inputs in the constructor
    def __init__(self, digit_classification_model, comparison_model):
        super(Full_Net_Hot, self).__init__() # I don't know the purpose of this line but i put it
        self.digit_classification_model=digit_classification_model
        self.comparison_model=comparison_model
    def forward(self, x):
        image1, image2 =split_images(x)
        x1=self.digit_classification_model(image1) #returns a tensor of size n x 10
        x2=self.digit_classification_model(image2) #returns a tensor of size n x 10
        # we want to get n x 2 with the index
        x =  torch.cat((x1, x2), dim=1)
        return self.comparison_model(x) # Might need to put a torch.cat((x1, x2), dim=0) here to feed the comparison model...

def train_model(model, train_input, train_target, test_input, test_target, batch_size=100, epochs=150, lr = 0.01, printing = True, full = False, auxiliaryLoss = 0.2):  # TODO: implement smart learning rate
    # for the full net target is a tuple containing digit1, digit2, comparison
    criterion = torch.nn.CrossEntropyLoss() #Compare w/ softmargin loss

    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range (0, epochs):
        sum_loss = 0

        for batch in range(0, train_input.size(0), batch_size): # Check out these functions, the sizes dont match: 25 & 100
            mini_batch = train_input.narrow(0, batch, batch_size)
            if not full:
                loss = criterion(model(mini_batch), train_target.narrow(0, batch, batch_size).flatten().long()) #might need to flatten
            else:
                digit1_hot, digit2_hot, comparison = model(mini_batch)
                loss = auxiliaryLoss * criterion(digit1_hot, train_target[0].narrow(0, batch, batch_size).flatten().long()) + auxiliaryLoss * criterion(digit2_hot, train_target[1].narrow(0, batch, batch_size).flatten().long()) + (1-2 * auxiliaryLoss) * criterion(comparison, train_target[2].narrow(0, batch, batch_size).flatten().long())
            sum_loss += loss.item() # item = to digit.
            model.zero_grad() #What does this do again?
            loss.backward() #What does this do again?
            optimizer.step() #includes model.train
        with torch.no_grad():
            if printing == True:
                if not full:
                    print('epoch {:d} train error: {:0.2f}% test error: {:0.2f}%'.format(epoch, compute_nb_errors(model, train_input, train_target, batch_size) / train_input.size(0) * 100, compute_nb_errors(model, test_input, test_target, batch_size) / test_input.size(0) * 100))
                else:
                    size_train = train_input.size()[0]
                    size_test = test_input.size()[0]
                    digit1_error, digit2_error, comparison_error = compute_nb_errors(model, train_input, train_target,
                                                                                       batch_size, full = True)
                    digit1_test_error, digit2_test_error, comparison_test_error = compute_nb_errors(model, test_input, test_target,
                batch_size, full = True)
                    print('epoch  {:d} train error: digit1 {:0.2f}, digit2 {:0.2f}%, comparison {:0.2f}'.format(epoch, digit1_error /size_train * 100, digit2_error / size_train * 100, comparison_error / size_train * 100))

                    print('epoch  {:d} test error: digit1 {:0.2f}, digit2 {:0.2f}%, comparison {:0.2f}'.format(epoch,
                                                                                                                digit1_test_error / size_test * 100,
                                                                                                                digit2_test_error / size_test * 100,
                                                                                                                comparison_test_error / size_test * 100))


def compute_nb_errors(model, data_input, data_target, mini_batch_size, full= False):

    nb_data_errors = 0
    nb_digit1_errors = 0
    nb_digit2_errors = 0
    nb_comparison_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if full == False:
            output = model(data_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(mini_batch_size):
                if data_target.data[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1
        if full == True:
            digit1, digit2, comparison = model(data_input.narrow(0, b, mini_batch_size))
            _, pred_digit1 = torch.max(digit1.data, 1)
            _, pred_digit2 = torch.max(digit2.data, 1)
            _, pred_comparison = torch.max(comparison.data, 1)

            for k in range(mini_batch_size):
                if data_target[0].data[b + k] != pred_digit1[k]:
                    nb_digit1_errors = nb_digit1_errors + 1
                if data_target[1].data[b + k] != pred_digit2[k]:
                    nb_digit2_errors = nb_digit2_errors + 1
                if data_target[2].data[b + k] != pred_comparison[k]:
                    nb_comparison_errors = nb_comparison_errors + 1


    if full == False:
        return nb_data_errors
    else:
        return nb_digit1_errors, nb_digit2_errors, nb_comparison_errors

def convert_to_hot(data_input):
    #converts 1D array of class indices (from 0 to 9) into 2D array with 1rst dimension denoting the example and second the dimension of size 20 with ones at the corresponding index.
    hot_data = torch.zeros(data_input.size()[0],20)
    hot_data[torch.arange(data_input.size()[0]), data_input[:,0]] = 1
    hot_data[torch.arange(data_input.size()[0]), data_input[:,1]+10] = 1

    return hot_data

def convert_hothot_to_digitdigit(hot1, hot2):
    #converts two hot vectors (nx10) to nx2 with the two numbers being the most probable digits of the hot vectors
    digit1 = torch.argmax(hot1, dim=1, keepdim=True).float()
    digit2 = torch.argmax(hot2, dim=1, keepdim=True).float()
    digitdigit = torch.cat((digit1, digit2), dim=1)
    return digitdigit

def split_TrainVal(input):
    N = input.size()[0]
    val = input.narrow(0,0,N//3)
    train = input.narrow(0, N//3, N-N//3)
    return train, val