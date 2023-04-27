
import paddle.nn as nn
import paddle
import numpy as np

np.random.seed(33)

# Define a simple neural network
class Net(nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 14 * 14, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
a = np.random.random(size=[1, 1, 28, 28])
# Create a random input tensor
x = paddle.to_tensor(a, dtype='float32')

# Initialize the neural network
net = Net()

# Compute the output of the network
y = net(x)

# Compute the gradient of the output with respect to the input
y.backward(paddle.ones_like(y))

# Check the gradient of the Conv2d layer's parameters
print(net.conv1.weight.grad)
print(net.conv1.bias.grad)



import torch
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

# Create a random input tensor
x = torch.tensor(a.astype(np.float32), requires_grad=True)

# Initialize the neural network
net = Net()

# Compute the output of the network
y = net(x)

# Compute the gradient of the output with respect to the input
y.backward(torch.ones_like(y))

# Check the gradient of the Conv2d layer's parameters
print(net.conv1.weight.grad)
print(net.conv1.bias.grad)
