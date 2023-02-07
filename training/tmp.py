
import torch
import torchvision
import numpy as np
train_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=lambda x: np.array(x)
)
import torch.utils.data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


tmp = next(iter(train_loader))
print(type(tmp))
print(type(tmp[0]))
print(type(tmp[1]))
