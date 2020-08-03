'''
The main training script where the actual training is done.
'''

##Import statements.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import MNISTNet
from mnist_data import MNISTData

import argparse
import os
import sys

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    ## Training code goes here.
    #model.train()
    iteration = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ## optimizer, run forward, calculate loss, backward.

        if iteration % 10 == 0:
            print ("Train epoch: {}, iteration: {}, training loss: {: .2f}".format(epoch, iteration, loss.item()))
        iteration += 1

def test(model, device, test_loader, criterion):
    ## Test or inference code goes here.
    #model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ## Run forward, calculate loss, predict values.
            # Fill here.
            
            pred = output.argmax(dim=1, keep_dim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print ("Test set: Average Loss: {}, Accuracy: {}".format(test_loss, 100.0 * (correct/len(test_loader.dataset))))

def main():
    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Get dataset handle and train and test dataloader.
    #code goes here.
    
    #get the model.
    model = MNISTNet().to(device)
    #Create optimizer
    #optimizer = 
    #criterion
    criterion = nn.CrossEntropyLoss()
    
    ## Train and test the network.
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    ## Save the network.
    if args.save_model:
        ## Code goes here.
    

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, required=False, default=64, help="Batch size of the training data")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Number of epochs for training model.")
    parser.add_argument('--save-model', action='store_true', default=False, help="Save the model in onnx format.")
    parser.add_argument('--use-cpu', action='store_true', default=False, help="Use CPU for training.")

    args = parser.parse_args()
    main()
