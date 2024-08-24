import torch
import torch.nn as nn


class Hypernetwork(nn.Module):
    def __init__(
        self,
        n_qubits,
        n_layers,
        n_hamiltonian_rep_dim,
        n_output_dim_each_layer,
        mlp_hidden_dim=64,
    ):
        """
        Args:
            n_hidden (int): 隐空间表示的维度
            n_layers (int): 目标网络的层数
            mlp_hidden_dim (int): MLP隐藏层的维度
        """
        super(Hypernetwork, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_output_dim_each_layer = n_output_dim_each_layer

        # 两层MLP用于初步处理拼接的隐空间表示
        self.initial_mlp = nn.Sequential(
            nn.Linear(n_hamiltonian_rep_dim, mlp_hidden_dim),  # 假设有10个节点
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        # 为每一层目标网络定义一个两层的MLP
        self.mlp_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden_dim, self.n_output_dim_each_layer),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, batch_z):
        # 拼接所有节点的隐空间表示
        z_0 = torch.reshape(batch_z, (batch_z.size(0), -1))
        # 通过两层MLP的初步处理
        z_1 = self.initial_mlp(z_0)

        # 为每一层目标网络生成参数
        generated_layer_params = []
        for idx, mlp in enumerate(self.mlp_list):
            generated_params = mlp(z_1)
            generated_layer_params.append(generated_params)
            # 输出第一层参数用于调试
            # if idx == 0:
            #     print("First layer parameters: ", params)

        # 将每一层的参数组合成一个整体
        generated_layer_params = torch.stack(generated_layer_params)
        generated_layer_params = generated_layer_params.permute(
            1, 2, 0
        )  # (Layer, Batch, n_output_dim_each_layer) -> (Batch, n_output_dim_each_layer, Layer)
        return generated_layer_params
