# Code generally copied from https://www.youtube.com/watch?v=gBw0u_5u0qU
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the data
x, y = torch.load("Data/MNIST/processed/training.pt")
print('finished loading data')


# Use a Dataset object
class CTDataset(Dataset):
    def __init__(self, filepath, n=0):
        self.x, self.y = torch.load(filepath)
        if n:
            self.x, self.y = self.x[:n], self.y[:n]
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


# Get the datasets and the dataloaders
train_ds = CTDataset('Data/MNIST/processed/training.pt')
test_ds = CTDataset('Data/MNIST/processed/test.pt')
train_dl = DataLoader(train_ds, batch_size=8)

train_acc_dl = DataLoader(CTDataset('Data/MNIST/processed/training.pt', n=1000), batch_size=8)
test_acc_dl = DataLoader(CTDataset('Data/MNIST/processed/test.pt'), batch_size=8)


# Define the neural net
class MyLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28 ** 2, 100)
        self.Matrix2 = nn.Linear(100, 3)
        self.Matrix3 = nn.Linear(3, 10)
        self.R = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 ** 2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 10),
            # 10
            nn.Softmax(),
            )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(10, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec


# Define accuracy
def accuracy(dl, f):
    num_correct = 0
    num_sampled = 0
    for x, y in dl:
        y_hats = f(x).argmax(axis=1)
        ys = y.argmax(axis=1)
        num_correct += torch.sum(y_hats == ys).item()
        num_sampled += len(y_hats)
    return num_correct / num_sampled


# Define training loop
default_epochs = 20


def train_model_linear(dl, f, n_epochs=default_epochs):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    train_accuracies = [accuracy(train_acc_dl, f)]
    test_accuracies = [accuracy(test_acc_dl, f)]
    for epoch in tqdm(range(n_epochs)):
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad()
            loss_value = L(f(x), y)
            loss_value.backward()
            opt.step()
            # Store training data
            epochs.append(epoch + i / N)
            losses.append(loss_value.item())
        train_accuracies.append(accuracy(train_acc_dl, f))
        test_accuracies.append(accuracy(test_acc_dl, f))
    return np.array(epochs), np.array(losses), train_accuracies, test_accuracies


def train_model_autoencoder(dl, f, n_epochs=default_epochs):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in tqdm(range(n_epochs)):
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad()
            loss_value = L(f(x), x.view(-1, 28 ** 2))
            loss_value.backward()
            opt.step()
            # Store training data
            epochs.append(epoch + i / N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses),


# Train a neural network
f = MyAutoencoder()
epoch_data, loss_data = train_model_autoencoder(train_dl, f)
epoch_data_avgd = epoch_data.reshape(20, -1).mean(axis=1)
loss_data_avgd = loss_data.reshape(20, -1).mean(axis=1)

# Plot the accuracies
# plt.plot(range(default_epochs + 1), train_accuracies, 'o--', label='Train Accuracy')
# plt.plot(range(default_epochs + 1), test_accuracies, 'o-', label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Train and Test Accuracy')
# plt.title('Train and Test Accuracy per Epoch')
# plt.legend()
# plt.show()

# Plot the loss
plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (avgd per epoch)')
plt.show()


# # Find the final accuracy on the test set
# print(f'Train accuracy: {train_accuracies[-1]}')
# print(f'Test accuracy: {test_accuracies[-1]}')

# Show some predictions
xs, ys = test_ds[:2000]
yhats = f(xs).view(-1, 28, 28)
fig, ax = plt.subplots(2, 10, figsize=(10, 15))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(xs[i])
    plt.subplot(2, 10, i + 11)
    plt.imshow(yhats[i].detach().numpy() )
fig.tight_layout()
plt.show()

# EMBEDDINGS ###########################################################################################################

# Store the embeddings for each of the different numbers, then find the average
def find_embeddings(dl, f):
    embeddings = defaultdict(list)
    for x, y in tqdm(dl):
        y_hat = f(x)
        # Compute it until the embedding layer
        x = x.view(-1, 28 ** 2)
        x = f.R(f.Matrix1(x))
        x = f.Matrix2(x)
        # Store the embeddings
        embeddings[y.argmax()].append(x)
        # Just for completeness, assert that the rest of the computation is executed correctly
        assert torch.equal(f.Matrix4(f.R(f.Matrix3(f.R(x) ) ) ).squeeze(), y_hat)
    return embeddings


# Find the actual embeddings
train_embedding_dl = DataLoader(CTDataset('Data/MNIST/processed/training.pt'), batch_size=1)
embeddings = find_embeddings(train_embedding_dl, f)
