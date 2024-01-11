import torch.nn as nn
import torch  
 
class MNISTMLPNeuralNetwork(nn.Module):
    def __init__(self):
        super(MNISTMLPNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        output = torch.softmax(x, dim=1)
        return output  