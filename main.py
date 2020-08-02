import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from model import MNISTNet
from mnist_data import MNISTData

import argparse
import os
import sys

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    iteration = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        #loss = -1.0 * F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print ("Train epoch : {}, iteration : {}, training loss : {: .2f}".format(epoch, iteration, loss.item()))
        iteration += 1

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            #test_loss += nn.CrossEntropyLoss(output, target, reduction='sum').item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print ("Test set : Average Loss : {}, Accuracy : {}".format(test_loss, 100.0 * (correct/len(test_loader.dataset))))

def main():
    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mnist_data_handle = MNISTData(args.batch_size, use_cuda)
    train_loader, test_loader = mnist_data_handle.get_train_and_test_dataloaders()
    
    model = MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
    
    if args.save_model:
        print ("INFO: Saving the model in onnx.")
        dummy_input = torch.randn(64, 1, 28, 28).to(device)
        torch.onnx.export(model, dummy_input, "mnist.onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, required=False, default=64, help="Batch size required for training.")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Number of epochs to train.")
    parser.add_argument('--save-model', action='store_true', default=False, help="Save the model in ONNX format")
    parser.add_argument('--use-cpu', action='store_true', default=False, help="Use CPU based training")

    args = parser.parse_args()
    main()
