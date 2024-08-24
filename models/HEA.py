import torch
import torch.nn as nn
import pennylane as qml


class HEA(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(HEA, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # # 确定权重的形状
        # shape = qml.StronglyEntanglingLayers.shape(
        #     n_layers=self.n_layers, n_wires=self.n_qubits
        # )

        # # 创建一个随机的权重张量，并将其转换为可训练的参数
        # self.params = torch.nn.Parameter(torch.rand(shape))

    def quantum_circuit(self, params, hamiltonian):
        """Define the quantum circuit and return the expectation value of the Hamiltonian."""

        @qml.qnode(self.dev, interface="torch")
        # @qml.qnode(self.dev, interface="numpy")
        def circuit(params, hamiltonian):
            qml.StronglyEntanglingLayers(
                weights=params, wires=range(self.n_qubits), imprimitive=qml.CNOT
            )
            return qml.expval(hamiltonian)

        return circuit(params, hamiltonian)

    def forward(self, batch_params, batch_hamiltonian):

        quantum_outputs = torch.stack(
            [
                self.quantum_circuit(params, hamiltonian)
                for params, hamiltonian in zip(batch_params, batch_hamiltonian)
            ]
        )

        return quantum_outputs
