import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(MLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_encoder_output = hidden_dims[-1]

        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class MLPDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dims):
        super(MLPDecoder, self).__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.decoder = nn.Sequential(*layers)
        print(self)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.n_encoder_output = encoder_hidden_dims[-1]

        self.encoder = MLPEncoder(input_dim, encoder_hidden_dims)
        self.decoder = MLPDecoder(input_dim, decoder_hidden_dims)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
