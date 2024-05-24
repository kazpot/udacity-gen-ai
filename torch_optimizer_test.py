import torch.optim as optim

"""
Gradients: Directions and amounts by which a function increases most.
The parameters can be changed in a direction opposite to the gradient of the loss
function in order to reduce the loss.

Learning Rate: This hyperparameter specifies how big the steps are when adjusting 
the neural network's settings during training. Too big, and you might skip over 
the best setting; too small, and it'll take a very long time to get there.

Momentum: A technique that helps accelerate the optimizer in the right direction 
and dampens oscillations.
"""

# Assuming model is your defined neural network
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# lr=0.01 sets the learning rate to 0.01
# momentum=0.9 smooths out updates and can help training