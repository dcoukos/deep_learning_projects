import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as dlc


# TODO: niel's line
train_input, train_target, train_classes, \
test_input, test_target, test_classes = dlc.generate_pair_sets(1000)


class Parallel_Net(nn.Module):
    def __init__(self):
        super(Parallel_Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3) # Compare 1 & 2 input neurons
        # Remember that kernel size decides how many pixels you lose. kernel =3 --> 14 px -> 12 px
        # Size of conv2d tells you input layers and output layers.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc1 = nn.Linear(64, 200)
        self.fc2 = nn.Linear(200, 1)
        # self.out = nn.Linear(20, 1) #TODO: test w/ addtional Output Layer

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        print(x.size())
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = F.relu(self.fc2(x.view(-1, 200)))
        x = torch.tanh(x)
        return x


train_input, train_target = Variable(train_input), Variable(train_target)

model = Parallel_Net()

def train_model(epochs=150, batch_size=100, lr=0.1):  # TODO: implement smart learning rate
    criterion = torch.nn.SoftMarginLoss() #Compare w/ softmargin loss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range (0, epochs):
        sum_loss = 0

        for batch in range(0, train_input.size(0), batch_size): # Check out these functions, the sizes dont match: 25 & 100
            mini_batch = train_input.narrow(0, batch, batch_size)
            loss = criterion(model(mini_batch).flatten(),
                             train_target.narrow(0, batch, batch_size).float()) #might need to flatten
            sum_loss += loss.item() # item = to digit.
            model.zero_grad() #What does this do again?
            loss.backward() #What does this do again?
            optimizer.step() #includes model.train

train_model(batch_size=1000)



mini_batch = train_input.narrow(0, 0, 1000)
model(mini_batch).size()
