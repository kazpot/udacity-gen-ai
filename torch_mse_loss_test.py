import torch
import torch.nn as nn

"""
Mean squared error: This shows the average of the squares of the differences
between predicted numbers and actual numbers.
It's often used for predicting continuous values rather than categories.
"""

# Define the loss function
loss_function = nn.MSELoss()

# Define the predicted and actual vlues as tensors
predicted_tensor = torch.tensor([[320000.0]])
actual_tensor = torch.tensor([300000.0])

loss_value = loss_function(predicted_tensor, actual_tensor)
print(loss_value.item())