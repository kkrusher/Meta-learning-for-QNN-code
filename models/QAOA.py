import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

from pennylane import qaoa


class QAOA(nn.Module):
    def __init__(self, n_qubits, p, cost_hamiltonian, mixer_hamiltonian=None):
        super(QAOA, self).__init__()
        self.n_qubits = n_qubits
        self.p = p
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.cost_hamiltonian = cost_hamiltonian

        if mixer_hamiltonian is None:
            self.mixer_hamiltonian = qml.Hamiltonian(
                [1] * n_qubits, [qml.PauliX(i) for i in range(n_qubits)]
            )
        else:
            self.mixer_hamiltonian = mixer_hamiltonian

    def qaoa_layer(self, gamma, alpha):
        qaoa.cost_layer(gamma, self.cost_hamiltonian)
        qaoa.mixer_layer(alpha, self.mixer_hamiltonian)
        # qaoa.cost_layer(gamma, self.cost_hamiltonian, wires=range(self.n_qubits))
        # qaoa.mixer_layer(alpha, self.mixer_hamiltonian, wires=range(self.n_qubits))

    def forward(self, params_gamma, params_alpha):
        @qml.qnode(self.dev, interface="torch")
        def circuit(params_gamma, params_alpha):
            qml.layer(self.qaoa_layer, self.p, params_gamma, params_alpha)
            return qml.expval(self.cost_hamiltonian)

        return circuit(params_gamma, params_alpha)
