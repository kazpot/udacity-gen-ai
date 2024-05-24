import torch
import matplotlib.pyplot as plt

# Create a 3-dimensional tensor
images = torch.rand((4, 28, 28))

second_image = images[1]

plt.imshow(second_image, cmap="gray")
plt.axis('off')
plt.show()
