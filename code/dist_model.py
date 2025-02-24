import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc11 = nn.Linear(input_dim, hidden_dim1)
        self.fc12 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1*2, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x1, x2):
        layer_11 = F.relu(self.fc11(x1))
        layer_12 = F.relu(self.fc12(x2))
        layer_1 = torch.cat((layer_11, layer_12), -1)
        layer_2 = F.relu(self.fc2(layer_1))
        layer_3 = F.relu(self.fc3(layer_2))
        out_layer = torch.sigmoid(self.fc4(layer_3))
        return out_layer