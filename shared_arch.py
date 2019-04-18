import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as dlc


# TODO: niel's line
train_input, train_target, train_classes, \
test_input, test_target, test_classes = dlc.generate_pair_sets(1000)


class SharedWeight_Net(nn.Module):
    def __init__(self):
        super(SharedWeight_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1)) #14->12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)) # 6-> 4
        self.lin1 = nn.Linear(256,120)
        self. lin2 = nn.Linear(120,84)
        self.lin3 = nn.Linear(84,10)

        # self.out = nn.Linear(20, 1) #TODO: test w/ addtional Output Layer

    def forward(self, x):
        x = F.relu(self.conv1(x), kernel_size=2, stride=2, dilation = 1) #12->12
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2, dilation = 1)) #4 -> 2
        print(x.size())
        x = F.relu(self.lin1(x.view(-1, 256)))
        x = F.relu(self.lin2(x.view(-1, 120)))
        x = F.relu(self.lin3(x.view(-1, 84)))
        return x


train_input, train_target = Variable(train_input), Variable(train_target)

model = SharedWeight_Net()

def train_model(epochs=150, batch_size=100, lr=0.1):  # TODO: implement smart learning rate
    criterion = torch.nn.CrossEntropyLoss() #Compare w/ softmargin loss
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

train_model(batch_size=100)
