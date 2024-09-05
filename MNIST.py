# Code generally copied from https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(embedding_dim=50, epochs=20):
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

    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(28 ** 2, 50),
                nn.ReLU(True),
            )

            self.decoder = nn.Sequential(
                nn.Linear(50, 28 ** 2),
                nn.ReLU(True),
                nn.Tanh(),
            )

        def forward(self, x):
            enc = self.encoder(x)
            dec = self.decoder(enc)
            return dec


    def train_model_autoencoder(dl, f, n_epochs=epochs):
        # Optimization
        opt = Adam(f.parameters(), lr=0.005)
        L = nn.MSELoss()

        # Train model
        losses = []
        epochs = []
        for epoch in tqdm(range(n_epochs)):
            N = len(dl)
            for i, (x, y) in enumerate(dl):
                # Update the weights of the network
                opt.zero_grad()
                loss_value = L(f(x.view(-1, 28 ** 2) ), x.view(-1, 28 ** 2))
                loss_value.backward()
                opt.step()
                # Store training data
                epochs.append(epoch + i / N)
                losses.append(loss_value.item())
                # # DEBUG stuff
                # if not i:
                #     print(f"\\EPOCH is: {epoch}")
                #     print(f(x[0].view(-1, 28 ** 2) ) )
                #     print(x[0])
        return np.array(epochs), np.array(losses),


    # Train a neural network
    f = Autoencoder()
    epoch_data, loss_data = train_model_autoencoder(train_dl, f)
    epoch_data_avgd = epoch_data.reshape(20, -1).mean(axis=1)
    loss_data_avgd = loss_data.reshape(20, -1).mean(axis=1)

    # Plot the loss
    plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross Entropy')
    plt.title('Cross Entropy (avgd per epoch)')
    # plt.show()

    # f takes in a flat vector and splits out a flat vector
    # Show some predictions
    xs, ys = test_ds[:50]
    yhats = f(xs.view(-1, 28 ** 2) ).view(-1, 28, 28)
    fig, ax = plt.subplots(2, 10, figsize=(10, 15))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(xs[i])
        plt.subplot(2, 10, i + 11)
        plt.imshow(yhats[i].detach().numpy() )
    fig.tight_layout()
    # plt.show()

    # EMBEDDINGS ###########################################################################################################

    # Store the embeddings for each of the different numbers, then find the average
    def find_embeddings(dl, f):
        embeddings = defaultdict(list)
        for x, y in tqdm(dl):
            y_hat = f(x.view(-1, 28 ** 2) )
            # Compute it until the embedding layer
            x = x.view(-1, 28 ** 2)
            x = f.encoder(x)
            # Store the embeddings
            embeddings[y.argmax().item()].append(x.detach()[0] )
            # Just for completeness, assert that the rest of the computation is executed correctly
            assert torch.equal(f.decoder(x), y_hat)
        return embeddings


    # Find the actual embeddings
    train_embedding_dl = DataLoader(CTDataset('Data/MNIST/processed/training.pt', n=2000), batch_size=1)
    embeddings = find_embeddings(train_embedding_dl, f)
    average_embeddings = []
    for i in range(10):
        # print('Working on', i)
        value_list = embeddings[i]
        embedding_tensor = torch.stack(embeddings[i])
        mean_embedding_tensor = torch.mean(embedding_tensor, axis=0)
        average_embeddings.append(mean_embedding_tensor)
    average_embeddings = torch.stack(average_embeddings)

    # Plot the average embeddings
    averages = f.decoder(average_embeddings)
    fig, ax = plt.subplots(1, 10, figsize=(10, 3))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(averages[i].view(28, 28).detach().numpy() )
    fig.tight_layout()
    plt.savefig(f'Reconstructions/linear_{embedding_dim}dim_{epochs}epochs.png')


if __name__ == "__main__":
    main(embedding_dim=20, epochs=5)
    main(embedding_dim=30, epochs=5)
    main(embedding_dim=70, epochs=5)
