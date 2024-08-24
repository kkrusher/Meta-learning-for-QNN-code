import random
import torch
import pennylane as qml


def generate_hamiltonian(adjs):
    """
    Generate a single Hamiltonian matrix based on the adjacency matrices.

    Args:
    - adjs (list of torch.Tensor): List of adjacency matrices.

    Returns:
    - hamiltonian (torch.Tensor): The resulting Hamiltonian matrix.
    """
    num_nodes = adjs[0].shape[0]

    coeffs = []
    obs = []

    # Diagonal matrices for X, Y, Z transverse fields
    for i in range(3):
        diag_elements = adjs[i].diag()
        for wire in range(num_nodes):
            if diag_elements[wire].item() != 0:
                if i == 0:
                    obs.append(qml.PauliX(wire))
                elif i == 1:
                    obs.append(qml.PauliY(wire))
                else:
                    obs.append(qml.PauliZ(wire))
                coeffs.append(diag_elements[wire].item())

    # Symmetric matrices for XX, YY, ZZ interactions
    for i in range(3, 6):
        for j in range(num_nodes):
            for k in range(j + 1, num_nodes):
                coeff = adjs[i][j, k].item()
                if coeff != 0:
                    if i == 3:
                        obs.append(qml.PauliX(j) @ qml.PauliX(k))
                    elif i == 4:
                        obs.append(qml.PauliY(j) @ qml.PauliY(k))
                    else:
                        obs.append(qml.PauliZ(j) @ qml.PauliZ(k))
                    coeffs.append(coeff)

    # Non-diagonal matrices for XY, YZ, ZX interactions
    for i in range(6, 9):
        for j in range(num_nodes):
            for k in range(num_nodes):
                coeff = adjs[i][j, k].item()
                if coeff != 0:
                    if i == 6:
                        obs.append(qml.PauliX(j) @ qml.PauliY(k))
                    elif i == 7:
                        obs.append(qml.PauliY(j) @ qml.PauliZ(k))
                    else:
                        obs.append(qml.PauliZ(j) @ qml.PauliX(k))
                    coeffs.append(coeff)

    # Combine coefficients and observables into a single Hamiltonian
    hamiltonian = qml.Hamiltonian(coeffs, obs)
    h_matrix = hamiltonian.sparse_matrix().toarray()

    return hamiltonian, h_matrix


def generate_adj_matrices(
    num_nodes,
    hamiltonian_model_type,
    scale=1.0,
    has_transverse_field=False,
    num_relations=9,
):
    """
    Generate adjacency matrices according to specific rules based on the model type.
    0-2 are diagonal matrices for transverse fields X, Y, Z
    3-5 are symmetric matrices with zero diagonal for XX, YY, ZZ interactions
    6-8 are non-diagonal matrices for XY, YZ, ZX interactions

    Args:
    - num_nodes (int): Number of nodes in the graph.
    - hamiltonian_model_type (str): The type of Hamiltonian to generate.
    - a (float): Range for the uniform distribution [-a, a].
    - num_relations (int): Number of types of relations (edges), fixed to 9.

    Returns:
    - adjs (list of torch.Tensor): List of adjacency matrices, one per relation.
    """
    # Initialize with zero matrices, to be filled based on hamiltonian_model_type
    adjs = torch.zeros((num_relations, num_nodes, num_nodes))

    def uniform_random():
        # Generate a random number uniformly distributed in the range [-a, a]
        return (2 * scale) * torch.rand(1).item() - scale

    if hamiltonian_model_type == "Ising":
        # Ising model: only ZZ interaction, symmetrically between adjacent nodes
        zz_matrix = torch.zeros((num_nodes, num_nodes))
        J = uniform_random()
        for i in range(num_nodes):
            zz_matrix[i, (i + 1) % num_nodes] = J
        zz_matrix = zz_matrix + zz_matrix.T  # Make the matrix symmetric
        adjs[5] = zz_matrix  # Only ZZ (6th matrix) has values, others are zeros

    elif hamiltonian_model_type == "SK":
        # SK model: only ZZ interaction, symmetrically between adjacent nodes
        zz_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            zz_matrix[i, (i + 1) % num_nodes] = uniform_random()
        zz_matrix = zz_matrix + zz_matrix.T  # Make the matrix symmetric
        adjs[5] = zz_matrix  # Only ZZ (6th matrix) has values, others are zeros

    elif hamiltonian_model_type == "XY":
        # XY model: only XX and YY interactions, symmetrically between adjacent nodes
        xx_matrix = torch.zeros((num_nodes, num_nodes))
        yy_matrix = torch.zeros((num_nodes, num_nodes))
        J = uniform_random()
        for i in range(num_nodes):
            xx_matrix[i, (i + 1) % num_nodes] = J
            yy_matrix[i, (i + 1) % num_nodes] = J
        xx_matrix = xx_matrix + xx_matrix.T  # Make symmetric
        yy_matrix = yy_matrix + yy_matrix.T  # Make symmetric
        adjs[3] = xx_matrix  # XX interaction (4th matrix)
        adjs[4] = yy_matrix  # YY interaction (5th matrix)

    elif hamiltonian_model_type == "XXX":
        # for XXX model: XX, YY, and ZZ interactions, symmetrically between adjacent nodes
        xx_matrix = torch.zeros((num_nodes, num_nodes))
        yy_matrix = torch.zeros((num_nodes, num_nodes))
        zz_matrix = torch.zeros((num_nodes, num_nodes))
        J = uniform_random()
        for i in range(num_nodes):
            xx_matrix[i, (i + 1) % num_nodes] = J
            yy_matrix[i, (i + 1) % num_nodes] = J
            zz_matrix[i, (i + 1) % num_nodes] = J
        xx_matrix = xx_matrix + xx_matrix.T  # Make symmetric
        yy_matrix = yy_matrix + yy_matrix.T  # Make symmetric
        zz_matrix = zz_matrix + zz_matrix.T  # Make symmetric
        adjs[3] = xx_matrix  # XX interaction (4th matrix)
        adjs[4] = yy_matrix  # YY interaction (5th matrix)
        adjs[5] = zz_matrix  # ZZ interaction (6th matrix)

    elif hamiltonian_model_type == "XXZ":
        # XXZ model: XX, YY, and ZZ interactions, symmetrically between adjacent nodes
        xx_matrix = torch.zeros((num_nodes, num_nodes))
        yy_matrix = torch.zeros((num_nodes, num_nodes))
        zz_matrix = torch.zeros((num_nodes, num_nodes))
        J = uniform_random()
        J_z = uniform_random()
        for i in range(num_nodes):
            xx_matrix[i, (i + 1) % num_nodes] = J
            yy_matrix[i, (i + 1) % num_nodes] = J
            zz_matrix[i, (i + 1) % num_nodes] = J_z
        xx_matrix = xx_matrix + xx_matrix.T  # Make symmetric
        yy_matrix = yy_matrix + yy_matrix.T  # Make symmetric
        zz_matrix = zz_matrix + zz_matrix.T  # Make symmetric
        adjs[3] = xx_matrix  # XX interaction (4th matrix)
        adjs[4] = yy_matrix  # YY interaction (5th matrix)
        adjs[5] = zz_matrix  # ZZ interaction (6th matrix)

    elif hamiltonian_model_type == "XYZ":
        # XYZ model: XX, YY, and ZZ interactions, symmetrically between adjacent nodes
        xx_matrix = torch.zeros((num_nodes, num_nodes))
        yy_matrix = torch.zeros((num_nodes, num_nodes))
        zz_matrix = torch.zeros((num_nodes, num_nodes))
        J_x = uniform_random()
        J_y = uniform_random()
        J_z = uniform_random()
        for i in range(num_nodes):
            xx_matrix[i, (i + 1) % num_nodes] = J_x
            yy_matrix[i, (i + 1) % num_nodes] = J_y
            zz_matrix[i, (i + 1) % num_nodes] = J_z
        xx_matrix = xx_matrix + xx_matrix.T  # Make symmetric
        yy_matrix = yy_matrix + yy_matrix.T  # Make symmetric
        zz_matrix = zz_matrix + zz_matrix.T  # Make symmetric
        adjs[3] = xx_matrix  # XX interaction (4th matrix)
        adjs[4] = yy_matrix  # YY interaction (5th matrix)
        adjs[5] = zz_matrix  # ZZ interaction (6th matrix)

    elif hamiltonian_model_type == "HM-I":
        # Inhomogeneous Heisenberg model: inhomogeneous values for XX, YY, ZZ interactions
        xx_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            J_ij = uniform_random()
            xx_matrix[i, (i + 1) % num_nodes] = J_ij
        xx_matrix = xx_matrix + xx_matrix.T  # Make symmetric
        adjs[3] = xx_matrix  # XX interaction (4th matrix)
        adjs[4] = xx_matrix  # YY interaction (5th matrix)
        adjs[5] = xx_matrix  # ZZ interaction (6th matrix)

    elif hamiltonian_model_type == "HM-L":
        # Long-range Heisenberg model: all elements of XX, YY, ZZ matrices may be non-zero
        J = uniform_random()
        xx_matrix = J * torch.ones((num_nodes, num_nodes))
        xx_matrix.fill_diagonal_(0)
        for i in range(3, 6):  # XX YY ZZ interaction
            adjs[i] = xx_matrix

    elif hamiltonian_model_type == "HM-O":
        # Heisenberg model with off-diagonal terms: allow XY, YZ, ZX interactions
        J = uniform_random()
        xx_matrix = J * torch.ones((num_nodes, num_nodes))
        xx_matrix.fill_diagonal_(0)
        for i in range(3, 9):  # XX YY ZZ interaction
            adjs[i] = xx_matrix

    elif hamiltonian_model_type == "HM-AILO":
        # Generate symmetric matrices with zero diagonal for relations 4-6 (XX, YY, ZZ interactions)
        for i in range(3):
            adj = (2 * scale) * torch.rand(num_nodes, num_nodes) - scale
            adj = (adj + adj.T) / 2  # Make the matrix symmetric
            adj.fill_diagonal_(0)  # Set diagonal elements to 0
            adjs[i + 3] = adj

        # Generate non-diagonal matrices for relations 7-9 (XY, YZ, ZX interactions)
        for i in range(3):
            adj = (2 * scale) * torch.rand(num_nodes, num_nodes) - scale
            adj.fill_diagonal_(0)  # Set diagonal elements to 0
            adjs[i + 6] = adj

    else:
        raise ValueError("Invalid model type")

    if has_transverse_field:
        J = uniform_random()
        adjs[2].fill_diagonal_(J)  # transverse Z field

    return adjs


def generate_random_data(
    num_nodes,
    hamiltonian_model_types=None,
    hamiltonian_model_type_weights=None,
    has_transverse_field=None,
):
    """
    Generate random node features and adjacency matrices.

    Args:
    - num_nodes (int): Number of nodes in the graph.
    - hamiltonian_model_types (list of str): List of Hamiltonian types to generate.
    - hamiltonian_model_type_weights (list of float): Weights for each Hamiltonian type.
    - has_transverse_field (bool): Whether to include a transverse field in the model.

    Returns:
    - selected_type (str): The selected Hamiltonian type.
    - x (torch.Tensor): Node feature matrix of shape (num_nodes, num_features).
    - adjs (list of torch.Tensor): List of adjacency matrices, one per relation.
    - hamiltonian (torch.Tensor): The resulting Hamiltonian matrix.
    - adjs_flattened (torch.Tensor): Flattened and concatenated adjacency matrices.
    """
    # Generate one-hot encoded node features
    x = torch.eye(num_nodes)

    if hamiltonian_model_types is None:
        hamiltonian_model_types = [
            "Ising",
            "SK",
            "XY",
            "XXX",
            "XXZ",
            "XYZ",
            "HM-I",
            "HM-L",
            "HM-O",
            "HM-AILO",
        ]

    # Normalize weights
    if hamiltonian_model_type_weights is None:
        hamiltonian_model_type_weights = [1.0] * len(hamiltonian_model_types)
    total_weight = sum(hamiltonian_model_type_weights)
    normalized_weights = [w / total_weight for w in hamiltonian_model_type_weights]

    # Set has_transverse_field to a random True or False if it is None
    if has_transverse_field is None:
        has_transverse_field = random.choice([True, False])

    # Select Hamiltonian type based on weights
    selected_type = random.choices(hamiltonian_model_types, weights=normalized_weights)[
        0
    ]

    # Generate adjacency matrices based on the selected Hamiltonian type
    adjs = generate_adj_matrices(
        num_nodes, selected_type, has_transverse_field=has_transverse_field
    )

    # Generate Hamiltonian matrix
    hamiltonian, _ = generate_hamiltonian(adjs)

    # Extract and concatenate specific elements from the adjacency matrices
    adjs_flattened = extract_specific_elements_from_adjs(adjs)

    assert (
        len(adjs_flattened) == (3 + 3 / 2) * num_nodes * (num_nodes - 1) + 1
    ), "extract specific elements from adjs, but got error length %d" % len(
        adjs_flattened
    )

    return selected_type, x, adjs, hamiltonian, adjs_flattened


def extract_specific_elements_from_adjs(adjs):
    """
    Extract specific elements from the adjacency matrices:
    - First element of the diagonal from the 3rd matrix.
    - Upper triangular part (excluding diagonal) of the 4th to 6th matrices.
    - All elements from the 7th to 9th matrices excluding the diagonal.

    Args:
    - adjs (list of torch.Tensor): List of adjacency matrices.

    Returns:
    - adjs_flattened (torch.Tensor): Concatenated 1D tensor with specific elements.
    """
    selected_elements = []

    # Extract the first element of the diagonal from the 3rd matrix
    # 包含长度为 1的横向场项
    selected_elements.append(adjs[2].diagonal()[0].unsqueeze(0))

    # Extract the upper triangular part (excluding diagonal) from the 4th to 6th matrices
    # 包含长度为 3n(n-1)/2 的xx yy zz 项
    for i in range(3, 6):
        upper_triangular_indices = torch.triu_indices(
            adjs[i].size(0), adjs[i].size(1), offset=1
        )
        selected_elements.append(
            adjs[i][upper_triangular_indices[0], upper_triangular_indices[1]]
        )

    # Extract all elements excluding the diagonal from the 7th to 9th matrices
    # 包含长度为 3n(n-1) 的xy yz zx 项
    for i in range(6, 9):
        off_diagonal_indices = torch.triu_indices(
            adjs[i].size(0), adjs[i].size(1), offset=1
        )
        selected_elements.append(
            adjs[i][off_diagonal_indices[0], off_diagonal_indices[1]]
        )
        off_diagonal_indices_lower = torch.tril_indices(
            adjs[i].size(0), adjs[i].size(1), offset=-1
        )
        selected_elements.append(
            adjs[i][off_diagonal_indices_lower[0], off_diagonal_indices_lower[1]]
        )

    # Concatenate all selected elements into a single 1D tensor
    adjs_flattened = torch.cat(selected_elements)

    return adjs_flattened
