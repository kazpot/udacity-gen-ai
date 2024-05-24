import torch
import torch.nn as nn

"""
Cross entropy loss: This is a measure used when a model needs to choose between categories
and it shows how well the model's predictions align with the actual categories.
"""

loss_function = nn.CrossEntropyLoss()

# Our dataset contains a single image of a dog, where
# cat = 0 and dog = 1 (corresponding to index 0 and 1)
target_tensor = torch.tensor([1])
print(target_tensor)

# Prediction: Most likely a dog (index 1 is higher)
# Note that the vlaues do not need to sum to 1
predicted_tensor = torch.tensor([[2.0, 5.0]])
loss_value = loss_function(predicted_tensor, target_tensor)
print(loss_value)

# Prediction: Most likely a cat (index 0 is higher)
predicted_tensor = torch.tensor([[1.5, 1.1]])
loss_value = loss_function(predicted_tensor, target_tensor)
print(loss_value)