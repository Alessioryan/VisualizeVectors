import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 ** 2, embedding_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 28 ** 2),
            nn.ReLU(True),
            nn.Tanh(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec