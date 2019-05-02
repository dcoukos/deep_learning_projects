import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F





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

class Comparison_Net_Hot(nn.Module):
    # this module takes as input a hot vector of size 20 (with all zeros and 1 at the place of the correct class) from the shared_weight net
    # and returns two activations (that will correspond to "bigger" neuron or to "smaller or/equal" neuron)
    def __init__(self):
        super(Comparison_Net_Hot, self).__init__()
        self.lin = nn.Linear(20,2)

    def forward(self, x):
        x = F.relu(self.lin(x.view(-1, 20)))
        return x

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

def train_model(model, train_input, train_target, test_input, test_target, batch_size=100, epochs=150, lr = 0.01):  # TODO: implement smart learning rate
    criterion = torch.nn.CrossEntropyLoss() #Compare w/ softmargin loss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range (0, epochs):
        sum_loss = 0

        for batch in range(0, train_input.size(0), batch_size): # Check out these functions, the sizes dont match: 25 & 100
            mini_batch = train_input.narrow(0, batch, batch_size)
            # print(mini_batch.size())
            # print('lol1')
            # print((model(mini_batch)).size())
            # print('lol2')
            # print(train_target.narrow(0, batch, batch_size).flatten().size())
            # print('lol3')
            loss = criterion(model(mini_batch), train_target.narrow(0, batch, batch_size).flatten().long()) #might need to flatten
            sum_loss += loss.item() # item = to digit.
            model.zero_grad() #What does this do again?
            loss.backward() #What does this do again?
            optimizer.step() #includes model.train

        print('e {:d} error: {:0.2f}%'.format(epoch, compute_nb_errors(model, test_input, test_target, batch_size) / test_input.size(0) * 100))


def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

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