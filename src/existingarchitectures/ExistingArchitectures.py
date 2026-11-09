#!/usr/bin/env python
# coding: utf-8

# Exploration of untrained predefined CNN architectures (resnet, efficientnet) and their performance on this dataset.

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os


# In[2]:


torch.cuda.is_available()


# In[3]:


filepath = 'processed_dataset.pt'
loaded_data = torch.load(filepath)
loaded_features = loaded_data['features']
loaded_labels = loaded_data['labels']
loaded_labels = loaded_labels[:,0]


# In[7]:


class XrayImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.labels[idx]

wholedataset = XrayImageDataset(loaded_features, loaded_labels)

train_len = int(len(wholedataset) * 0.8)  # 80% for training
val_len = int(len(wholedataset) * 0.1)    # 10% for validation
test_len = len(wholedataset) - train_len - val_len # Remaining 10% for testing

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(wholedataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)


# In[9]:


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18()
# Replace the last layer with a custom one to output the correct number of classes
model.fc = nn.Linear(model.fc.in_features, 22)
model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[22]:


import time
num_epochs = 10
train_losses = []
train_accs = []
val_losses = []
val_accs = []
for epoch in range(num_epochs):
    start = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_loss, train_acc = validate(model, train_loader, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Time elapsed: {time.time() - start:.2f}s")


# In[23]:


torch.save(model.state_dict(), '10_resnet18.pth')


# In[24]:


num_epochs = 10

for epoch in range(num_epochs):
    start = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_loss, train_acc = validate(model, train_loader, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    
    print(f"Epoch {10+epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Time elapsed: {time.time() - start:.2f}s")


# In[25]:


torch.save(model.state_dict(), '20_RESNET18.pth')


# In[39]:


plt.plot(range(1,21),train_losses)
plt.plot(range(1,21), val_losses)
plt.legend(['Train','Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1,21))
plt.title("RESNET18 Loss during training")
plt.savefig("Resnet18Losses.png")


# In[38]:


plt.plot(range(1,21),train_accs)
plt.plot(range(1,21), val_accs)
plt.legend(['Train','Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1,21))
plt.title("RESNET18 Loss during training")
plt.savefig("Resnet18Accuracies.png")


# In[44]:


model = torchvision.models.efficientnet_b0()
model


# In[50]:


model = torchvision.models.efficientnet_b0()
# (features): Sequential(
#     (0): Conv2dNormActivation(
#       (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): SiLU(inplace=True)
#     )
model.features[0][0] = nn.Conv2d(1,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

# Replace the last layer with a custom one to output the correct number of classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 22)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[51]:


model


# In[52]:


num_epochs = 20
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    start = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_loss, train_acc = validate(model, train_loader, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Time elapsed: {time.time() - start:.2f}s")


# In[53]:


torch.save(model.state_dict(), "11_efficientnet.pth")


# In[54]:


num_epochs = 9

for epoch in range(num_epochs):
    start = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_loss, train_acc = validate(model, train_loader, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {11+epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Time elapsed: {time.time() - start:.2f}s")
torch.save(model.state_dict(), "20_efficientnet.pth")


# In[55]:


plt.plot(range(1,21),train_losses)
plt.plot(range(1,21), val_losses)
plt.legend(['Train','Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1,21))
plt.title("EfficientNet Loss during training")
plt.savefig("EfficientnetLosses.png")


# In[56]:


plt.plot(range(1,21),train_accs)
plt.plot(range(1,21), val_accs)
plt.legend(['Train','Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1,21))
plt.title("EfficientNet Loss during training")
plt.savefig("EfficientnetAccuracies.png")


# In[ ]:




