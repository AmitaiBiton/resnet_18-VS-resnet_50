from turtle import pd

import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import shutil
from torch.optim import lr_scheduler
import copy



def resize_image(src_image, size=(128,128), bg_color="white"):
    from PIL import Image, ImageOps

    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)

    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)

    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))

    return new_image


training_folder_name = './MRI_CT_data/all'

# New location for the resized images
train_folder = './DATA_OUT/'

# Create resized copies of all of the source images
size = (128  ,128)

# Create the output folder if it doesn't already exist
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)

for root, folders, files in os.walk(training_folder_name):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        # Create a  subfolder in the output location
        saveFolder = os.path.join(train_folder, sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        # Loop through  files in the subfolder (Open each  & resize & save
        file_names = os.listdir(os.path.join(root, sub_folder))
        for file_name in file_names:
            file_path = os.path.join(root, sub_folder, file_name)
            image = Image.open(file_path)
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            resized_image.save(saveAs)

def load_dataset(data_path):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    # Load all the images

    transformation = transforms.Compose([

        #torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0.5),

        #torchvision.transforms.FiveCrop(size),

        transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images and transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation)

    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1*len(full_dataset))
    test_size = (len(full_dataset) - train_size - val_size)

    train_dataset,val_dataset , test_dataset = torch.utils.data.random_split(full_dataset, [train_size,val_size, test_size])

    # training data , 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=70,
        num_workers=0,
        shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=30,
        num_workers=0,
        shuffle=False
    )
    #  testing data
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=30,
        num_workers=0,
        shuffle=False
    )

    return train_loader,val_loader, test_loader ,train_size ,val_size ,test_size

train_loader,val_loader , test_loader ,train_size ,val_size ,test_size = load_dataset(train_folder)
batch_size = train_loader.batch_size
print("Data loaders ready to read", train_folder)

# NetWork resnet18
model= models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs ,2)
print(model)

# NetWork resnet50
"""
model= models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2), 
    nn.LogSoftmax(dim=1)
)
"""
#loss_criterion = nn.NLLLoss()

device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda"
model.cuda()
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#73

def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset the optimizer
        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = loss_criterion(output, target)
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()
        # Print metrics so we see some progress
        #print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    #print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader  , string ):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            tmp = test_loss
            # Calculate the loss for this batch
            test_loss += loss_criterion(output, target).item()
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print( string,'set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # return average loss for the epoch
    return avg_loss , 100. * correct / len(test_loader.dataset)

torch.cuda.empty_cache()
# Train over 10 epochs (We restrict to 10 for time issues)

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)#70%
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#73%

Device = "cuda"
epochs = 10
print('Training on', device)
epoch_nums=[]
training_loss=[]
validation_loss=[]
accuracy_val = []

for epoch in range(1, epochs+1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    val_loss , accuracy_score_val = test(model, device, test_loader , "validation")
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
    accuracy_val.append(accuracy_score_val)

test_loss , accuracy_score_test = test(model, device, test_loader , "test")

plt.figure(figsize=(15,15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'  ,'test'], loc='upper right')
plt.show()

plt.plot(epoch_nums, accuracy_val)
plt.legend([ 'validation'], loc='upper right')
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")
plt.show()

