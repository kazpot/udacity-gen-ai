import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 64)
        self.ouput_layer = nn.Linear(64, 2)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        return self.ouput_layer(x)
    
model = MLP(input_size=10)
print(model)

print(model.forward(torch.rand(10)))