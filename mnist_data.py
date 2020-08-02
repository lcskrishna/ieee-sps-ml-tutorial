'''
In this file, you will learn how to use the dataloaders in pytorch.
1. Gather the MNIST Dataset from torchvision
2. Create a pre-processing transforms for MNIST data.
3. Create train loader and test loader.

'''

## Import statements.
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class MNISTData():
    def __init__(self, batch_size, use_cuda):
        super(MNISTData, self).__init__()
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def get_train_and_test_dataloaders(self):
        #Create transforms
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
        
        ## Gather the datasets.
        # Fill in the code here. 
        #train_dataset = 
        #test_dataset = 
        
        kwargs = {'batch_size' : self.batch_size}
        if self.use_cuda:
            kwargs.update({'num_workers': 1,
                           'pin_memory': True,
                           'shuffle' : True},)

        ##Create train loader and test loader for the datasets created above.
        #train_loader = 
        #test_loader = 

        #return train_loader, test_loader
