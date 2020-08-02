import torch
import torch.nn as nn
from torchvision import datasets, transforms

class MNISTData():
    def __init__(self, batch_size, use_cuda):
        super(MNISTData, self).__init__()
        print ("INFO: Creating the dataset.")
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def get_train_and_test_dataloaders(self):
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)) ])
        dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('data', train=False, download=True, transform=transform)

        kwargs = {'batch_size' : self.batch_size}
        if self.use_cuda:
            kwargs.update({'num_workers' : 1,
                           'pin_memory' : True,
                           'shuffle' : True}, )

        train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

        return train_loader, test_loader
