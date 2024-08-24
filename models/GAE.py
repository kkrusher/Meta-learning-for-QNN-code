import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_attention_scores(h, adj, att_layer):
    N = h.size(0)

    # Compute pair-wise attention scores
    a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
    a_input = a_input.view(N, -1, 2 * h.size(1))
    e = att_layer(a_input).squeeze(2)

    # Incorporate edge weights
    edge_weights = adj.clone()  # Clone adj matrix to represent edge weights
    zero_vec = -1e9 * torch.ones_like(e)
    attention = torch.where(adj > 0, e, zero_vec)  # Mask disconnected nodes

    # Multiply attention scores by edge weights
    attention = attention * edge_weights

    # Apply softmax to normalize
    attention = F.softmax(F.leaky_relu(attention), dim=1)

    return attention


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, n_relations=9):
        super(GCNLayer, self).__init__()

        assert n_relations == 9, "n_relations should be restricted to 9"
        self.n_relations = n_relations

        # for self-loop and undirected edges
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(6)]
        )
        self.att_list = nn.ModuleList(
            [nn.Linear(2 * out_features, 1) for _ in range(6)]
        )

        # for directed edges
        self.forward_fc_list = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(3)]
        )  # Forward weights for directed edges
        self.backward_fc_list = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(3)]
        )  # Backward weights for directed edges

        self.forward_att_list = nn.ModuleList(
            [nn.Linear(2 * out_features, 1) for _ in range(3)]
        )
        self.backward_att_list = nn.ModuleList(
            [nn.Linear(2 * out_features, 1) for _ in range(3)]
        )

        # Extra linear layer and bias for additional terms
        self.self_loop_fc = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.fc_list:
            nn.init.xavier_uniform_(fc.weight)
        for att in self.att_list:
            nn.init.xavier_uniform_(att.weight)
            nn.init.zeros_(att.bias)
        for fc in self.forward_fc_list:
            nn.init.xavier_uniform_(fc.weight)
        for fc in self.backward_fc_list:
            nn.init.xavier_uniform_(fc.weight)
        for att in self.forward_att_list:
            nn.init.xavier_uniform_(att.weight)
            nn.init.zeros_(att.bias)
        for att in self.backward_att_list:
            nn.init.xavier_uniform_(att.weight)
            nn.init.zeros_(att.bias)
        nn.init.xavier_uniform_(self.self_loop_fc.weight)
        nn.init.zeros_(self.bias)

    def forward(self, batch_x, batch_adjs):
        batch_output = []
        for x, adjs in zip(batch_x, batch_adjs):
            output = 0

            for i, adj in enumerate(adjs):
                if i < 3:
                    # Self-loop attention mechanism
                    h = self.fc_list[i](x)  # Linear transformation
                    attention = compute_attention_scores(h, adj, self.att_list[i])

                    # Aggregate the features
                    h_prime = torch.matmul(attention, h)
                    output += h_prime
                elif 3 <= i < 6:
                    # TODO 确认一下逻辑？
                    # Undirected edge attention mechanism
                    h = self.fc_list[i](x)

                    # Extract the diagonal elements and form a 1D vector
                    diag_elements = torch.diag(adj)
                    # Compute the outer product to form a full matrix
                    adj_self_loop = torch.outer(diag_elements, diag_elements)
                    # Compute the attention scores using the modified adjacency matrix
                    attention = compute_attention_scores(
                        h, adj_self_loop, self.att_list[i]
                    )

                    # Aggregate the features
                    h_prime = torch.matmul(attention, h)
                    output += h_prime
                else:
                    # Directed edge attention mechanism
                    forward_h = self.forward_fc_list[i - 6](x)
                    forward_attention = compute_attention_scores(
                        h, adj, self.forward_att_list[i - 6]
                    )
                    forward_h_prime = torch.matmul(forward_attention, forward_h)
                    output += forward_h_prime / 2

                    backward_h = self.backward_fc_list[i - 6](x)
                    backward_attention = compute_attention_scores(
                        h, adj.T, self.backward_att_list[i - 6]
                    )
                    backward_h_prime = torch.matmul(backward_attention, backward_h)
                    output += backward_h_prime / 2

            # Add the self-loop term and bias
            output += self.self_loop_fc(x) + self.bias

            # Apply ReLU activation and add to the list
            batch_output.append(F.relu(output))

        # Stack all the outputs to create a batch output tensor
        batch_output = torch.stack(batch_output)
        return batch_output


# GAE Encoder
class GAEEncoder(nn.Module):
    def __init__(
        self, n_qubits, n_features, n_hidden_list, n_encoder_output, n_relations=9
    ):
        super(GAEEncoder, self).__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_hidden_list = n_hidden_list
        self.n_encoder_output = n_encoder_output
        self.gcns = nn.ModuleList()
        self.fc = nn.Linear(n_hidden_list[-1] * n_qubits, n_encoder_output)

        # Create GCN layers according to the n_hidden_list list
        for i in range(len(n_hidden_list)):
            in_features = n_features if i == 0 else n_hidden_list[i - 1]
            out_features = n_hidden_list[i]
            self.gcns.append(GCNLayer(in_features, out_features, n_relations))

        logging.info(f"GAEEncoder input dim: {n_features}")
        logging.info(f"GAEEncoder output dim: {n_encoder_output}")

    def forward(self, x, adjs):
        hidden = x
        for gcn in self.gcns:
            hidden = gcn(hidden, adjs)

        # 将 hidden 拉平成二维 (batch_size, -1)
        hidden = torch.flatten(hidden, start_dim=1)

        output = self.fc(hidden)
        return output


# GAE Decoder
class GAEDecoder(nn.Module):
    def __init__(self, n_qubits, n_output, n_relations=9, mlp_hidden_dim=64):
        super(GAEDecoder, self).__init__()
        self.n_qubits = n_qubits
        self.n_output = n_output
        self.n_relations = n_relations
        self.mlp_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_output, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden_dim, self.n_qubits * self.n_qubits),
                )
                for _ in range(n_relations)
            ]
        )

    def forward(self, batch_z):
        batch_adj_reconstructed = []
        for z in batch_z:
            adj_reconstructed = []
            for i in range(self.n_relations):
                mlp = self.mlp_list[i]
                h_j_mlp = mlp(z).view(self.n_qubits, self.n_qubits)
                adj_reconstructed.append(h_j_mlp)
            batch_adj_reconstructed.append(adj_reconstructed)

        # Convert list of lists into a single tensor with shape (batch_size, n_relations, n_nodes, n_nodes)
        batch_adj_reconstructed = torch.stack(
            [
                torch.stack(adj_reconstructed)
                for adj_reconstructed in batch_adj_reconstructed
            ]
        )
        return batch_adj_reconstructed


class GAE(nn.Module):
    def __init__(
        self, n_qubits, n_features, n_hidden_list, n_encoder_output, n_relations=9
    ):
        super(GAE, self).__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_hidden_list = n_hidden_list
        self.n_encoder_output = n_encoder_output
        self.encoder = GAEEncoder(
            n_qubits, n_features, n_hidden_list, n_encoder_output, n_relations
        )
        self.decoder = GAEDecoder(n_qubits, n_encoder_output, n_relations)

    def forward(self, x, adjs):
        z = self.encoder(x, adjs)
        adj_reconstructed = self.decoder(z)
        return z, adj_reconstructed
