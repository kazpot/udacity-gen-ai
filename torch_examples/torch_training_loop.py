import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

"""
Training Loop: The cycle that a neural network goes through many times 
to learn from the data by making predictions, checking errors, and improving itself.

Batches: Batches are small, evenly divided parts of data 
that the AI looks at and learns from each step of the way.

Epochs: A complete pass through the entire training dataset. 
The more epochs, the more the computer goes over the material to learn.

Loss functions: They measure how well a model is performing 
by calculating the difference between the model's predictions and the actual results.

Optimizer: Part of the neural network's brain that 
makes decisions on how to change the network to get better at its job.
"""

class NumberSumDataset(Dataset):
    def __init__(self, data_range=(1, 10)):
        self.numbers = list(range(data_range[0], data_range[1]))
    
    def __getitem__(self, index):
        number1 = float(self.numbers[index // len(self.numbers)])
        number2 = float(self.numbers[index % len(self.numbers)])
        return torch.tensor([number1, number2]), torch.tensor([number1 + number2])
    
    def __len__(self):
        return len(self.numbers) ** 2
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 128)
        self.output_layer = nn.Linear(128, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        return self.output_layer(x)

dataset = NumberSumDataset(data_range=(0, 100))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
model = MLP(input_size=2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for 10 epochs
for epoch in range(10):
    loss = 0.0
    for number_pairs, sums in dataloader: # Iterate over the batches
        predictions = model(number_pairs) # Compute the model output
        loss = loss_function(predictions, sums) # Compute the loss
        loss.backward() # Perform backpropagation
        optimizer.step() # Update the parameters
        optimizer.zero_grad() # Zero the gradients

        loss += loss.item() # Add the loss for all batches

    # Print the loss for this epoch
    print("Epoch {}: Sum of Batch Losses = {:.5f}".format(epoch, loss))

# Test the model on 3 + 7
print(model(torch.tensor([3.0, 7.0])))