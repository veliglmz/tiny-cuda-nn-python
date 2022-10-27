import torch
import torch.nn as nn
from model_utils import determine_activation

torch.set_printoptions(precision=8)


class Network(nn.Module):
    def __init__(self, activation_name, n_input, n_neurons, n_hidden_layers, n_output):
        super().__init__()
        self.activation = determine_activation(activation_name)
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.n_output = n_output

        self.inputs = None  # we add this variable for differentiation.
        self.inputs_layer = nn.Linear(n_input, n_neurons, bias=False)
        self.hidden_layers = nn.Sequential()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons, bias=False))
            self.hidden_layers.append(self.activation)
        self.output_layer = nn.Linear(n_neurons, n_output, bias=False)

    def forward(self, x):
        self.inputs = x
        x = self.activation(self.inputs_layer(self.inputs))
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x[:, :3]