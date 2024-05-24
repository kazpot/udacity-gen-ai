from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
PyTorch Dataset class: This is like a recipe that tells your computer 
how to get the data it needs to learn from, including where to 
find it and how to parse it, if necessary.

PyTorch Data Loader: Think of this as a delivery truck that 
brings the data to your AI in small, manageable loads called batches; 
this makes it easier for the AI to process and learn from the data.

Batches: Batches are small, evenly divided parts of data that 
the AI looks at and learns from each step of the way.

Shuffle: It means mixing up the data so that it's not 
in the same order every time, which helps the AI learn better.
"""

# Create a toy dataset
class NumberProductDataset(Dataset):
    def __init__(self, data_range=(1, 10)):
        self.numbers = list(range(data_range[0], data_range[1]))
    
    def __getitem__(self, index):
        number1 = self.numbers[index]
        number2 = self.numbers[index] + 1
        return (number1, number2), number1 * number2
    
    def __len__(self):
        return len(self.numbers)

# Instantiate the dataset
dataset = NumberProductDataset(
    data_range=(0, 11)
)

# Create a DataLoader instance
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Iterating over batches
for (num_pairs, products) in dataloader:
    print(num_pairs, products)
