# Import necessary libraries
import argparse
import shutil
import json5
import os
import torch
import numpy as np
import random
import logging

from pathlib import Path
from argparse import Namespace

from utils.data_generator import generate_random_data
from models.QAOA import QAOA
from models.HEA import HEA


def generate_hidden_dims(start_dim, min_dim=100):
    hidden_dims = []
    current_dim = start_dim

    while current_dim >= min_dim:
        hidden_dims.append(current_dim)
        current_dim //= 2

    hidden_dims.append(min_dim)

    return hidden_dims


def set_log(exp_path):
    # Configure logging
    log_file_path = exp_path / "training_log.log"

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level

    # Clear any existing handlers from the root logger
    logger.handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler()

    # Set level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def set_random_seed(seed):
    # 设置Python内置的随机数生成器的种子
    random.seed(seed)

    # 设置numpy的随机数生成器的种子
    np.random.seed(seed)

    # 设置PyTorch的随机数生成器的种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设置PennyLane的随机数生成器的种子
    # qml.default.qubit.seed(seed)

    # 其他可能需要设置的全局随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_parser(config_file=None):
    parser = argparse.ArgumentParser(
        description="Train GAE and Hypernet for quantum architectures"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="HEA",
        choices=["HEA", "QAOA"],
        help="Architecture type",
    )
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--n_layers", type=int, default=10, help="Number of layers")

    parser.add_argument(
        "--ae_type",
        type=str,
        default="AE",
        choices=["AE", "GAE"],
        help="Autoencoder type",
    )
    # for GAE only
    # parser.add_argument(
    #     "--gnn_hidden_dim_list",
    #     type=int,
    #     nargs="+",
    #     default=[16, 10],
    #     help="List of hidden layer sizes for GNN",
    # )
    parser.add_argument(
        "--n_encoder_output",
        type=int,
        default=20,
        help="Output dimension for autoencoder",
    )

    parser.add_argument(
        "--hamiltonian_model_types",
        type=str,
        nargs="+",
        default=["Ising"],
        help="List of Hamiltonian model types for training",
    )
    parser.add_argument(
        "--hamiltonian_model_type_weights",
        type=float,
        nargs="+",
        default=[1.0],
        help="List of weights corresponding to the Hamiltonian model types for training",
    )
    parser.add_argument(
        "--test-hamiltonian_model_types",
        type=str,
        nargs="+",
        default=["Ising"],
        help="List of Hamiltonian model types for test",
    )
    parser.add_argument(
        "--test_hamiltonian_model_type_weights",
        type=float,
        nargs="+",
        default=[1.0],
        help="List of weights corresponding to the Hamiltonian model types for test",
    )
    parser.add_argument(
        "--has_transverse_field",
        type=bool,
        default=False,
        help="Whether the Hamiltonian has a transverse field",
    )
    parser.add_argument(
        "--gae_n_epochs",
        type=int,
        default=100,
        help="Number of epochs for GAE training",
    )
    parser.add_argument(
        "--gae_n_batches",
        type=int,
        default=100,
        help="Number of batches per epoch for GAE training",
    )
    parser.add_argument(
        "--gae_batch_size", type=int, default=8, help="Batch size for GAE training"
    )
    parser.add_argument(
        "--hypernet_n_epochs",
        type=int,
        default=100,
        help="Number of epochs for Hypernet training",
    )
    parser.add_argument(
        "--hypernet_n_batches",
        type=int,
        default=100,
        help="Number of batches per epoch for Hypernet training",
    )
    parser.add_argument(
        "--hypernet_batch_size",
        type=int,
        default=8,
        help="Batch size for Hypernet training",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to JSON file with configuration parameters",
    )
    parser.add_argument(
        "--n_train_from_scratch_steps",
        type=int,
        default=100,
        help="Number of steps to train a random qnn from scratch",
    )
    parser.add_argument(
        "--n_random_hamiltonian_tobe_train",
        type=int,
        default=5,
        help="Number of random hamiltonians to be trained from scratch",
    )
    args = parser.parse_args()

    if config_file is not None:
        args.config_file = config_file

    # 优先级是 命令行>json文件>默认值
    if args.config_file and os.path.isfile(args.config_file):
        with open(args.config_file, "r") as f:
            config_params = json5.load(f)
            for key, value in config_params.items():
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)

    # 校验hamiltonian_model_types和hamiltonian_model_type_weights的长度是否一致
    if len(args.hamiltonian_model_types) != len(args.hamiltonian_model_type_weights):
        raise ValueError(
            "The number of hamiltonian_model_types must match the number of hamiltonian_model_type_weights"
        )

    return args


def create_directory(path_input):
    """
    创建目录的函数。如果路径已存在，则抛出错误；如果路径不存在，则创建目录。

    参数:
    - path_input (str 或 Path): 要创建的目录路径，可以是字符串或 Path 对象。

    返回:
    - None
    """
    # 如果输入是字符串，则将其转换为 Path 对象
    if isinstance(path_input, str):
        path = Path(path_input)
    elif isinstance(path_input, Path):
        path = path_input
    else:
        raise TypeError("输入必须是字符串或 Path 对象")

    # 检查路径是否存在
    if path.exists():
        # # TODO 为了防止删除已有的实验结果，可以选择报错
        # raise FileExistsError(f"路径 '{path}' 已经存在。")
        try:
            # 删除已有的目录及其内容
            shutil.rmtree(path)
            logging.info(f"路径 '{path}' 已经存在，已被删除。")
        except Exception as e:
            raise RuntimeError(f"删除目录时发生错误: {e}")
    try:
        # 创建目录，包括必要的父目录
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"目录 '{path}' 已成功创建。")
    except Exception as e:
        raise RuntimeError(f"创建目录时发生错误: {e}")


def save_args_to_json(args, save_path):
    # Convert args to a dictionary
    args.exp_path = str(args.exp_path)
    args_dict = vars(args)

    # Define the path for the JSON file
    json_file_path = save_path / "args.json"

    # Save args to JSON
    with open(json_file_path, "w") as json_file:
        json5.dump(args_dict, json_file, indent=4)

    args.exp_path = Path(args.exp_path)


def load_args_from_json(json_file_path):
    # Load the JSON file into a dictionary
    with open(json_file_path, "r") as json_file:
        args_dict = json5.load(json_file)

    # Convert the dictionary back to a Namespace (or any other object type you use)
    args = Namespace(**args_dict)

    args.exp_path = Path(args.exp_path)

    return args


# Load pre-trained GAE and Hypernetwork models
def load_model(model_class, model_path, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


# Generate and evaluate a QAOA model using the trained Hypernetwork
def generate_and_evaluate_qaoa(
    n_qubits,
    n_layers,
    ae_type,
    ae_encoder,
    hypernet,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
):
    with torch.no_grad():
        # Generate data
        selected_type, x, adjs, hamiltonian, adjs_flattened = generate_random_data(
            n_qubits,
            hamiltonian_model_types=hamiltonian_model_types,
            hamiltonian_model_type_weights=hamiltonian_model_type_weights,
            has_transverse_field=has_transverse_field,
        )

        # Use GAE encoder to generate latent space representation
        batch_x = x.unsqueeze(
            0
        )  # x 的形状从 (num_nodes, num_features) 变为 (1, num_nodes, num_features)
        batch_adjs = adjs.unsqueeze(
            0
        )  # adjs 的形状从 (num_relations, num_nodes, num_nodes) 变为 (1, num_relations, num_nodes, num_nodes)
        batch_adjs_flattened = adjs_flattened.unsqueeze(0)

        if ae_type == "GAE":
            batch_z = ae_encoder(batch_x, batch_adjs)
        elif ae_type == "AE":
            batch_z = ae_encoder(batch_adjs_flattened)

        # Use Hypernetwork to generate QAOA parameters
        params = hypernet(batch_z).view(-1, 2, n_layers)

        # Initialize QAOA model
        qaoa = QAOA(n_qubits, n_layers, hamiltonian)
        qaoa.eval()

        # Forward propagate through QAOA model to compute the expectation value
        predicted_energy = qaoa(params[0][0], params[0][1])

        # Compute loss, the absolute value of the current scalar
        loss = torch.abs(torch.sum(predicted_energy)).item()

    return selected_type, adjs, hamiltonian, loss, (params[0][0], params[0][1])


# Generate and evaluate an HEA model using the trained Hypernetwork
def generate_and_evaluate_hea(
    n_qubits,
    n_layers,
    ae_type,
    ae_encoder,
    hypernet,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
):
    with torch.no_grad():
        # Generate data
        selected_type, x, adjs, hamiltonian, adjs_flattened = generate_random_data(
            n_qubits,
            hamiltonian_model_types=hamiltonian_model_types,
            hamiltonian_model_type_weights=hamiltonian_model_type_weights,
            has_transverse_field=has_transverse_field,
        )

        # # Use GAE encoder to generate latent space representation
        batch_x = x.unsqueeze(
            0
        )  # x 的形状从 (num_nodes, num_features) 变为 (1, num_nodes, num_features)
        batch_adjs = adjs.unsqueeze(
            0
        )  # adjs 的形状从 (num_relations, num_nodes, num_nodes) 变为 (1, num_relations, num_nodes, num_nodes)
        batch_adjs_flattened = adjs_flattened.unsqueeze(0)

        if ae_type == "GAE":
            batch_z = ae_encoder(batch_x, batch_adjs)
        elif ae_type == "AE":
            batch_z = ae_encoder(batch_adjs_flattened)

        # Use Hypernetwork to generate HEA parameters
        generated_params = hypernet(batch_z).view(-1, n_layers, n_qubits, 3)

        # Initialize HEA model
        hea = HEA(n_qubits, n_layers)
        hea.eval()

        # Forward propagate through HEA model to compute the expectation value
        predicted_energy = hea([generated_params], [hamiltonian])

        # Compute loss, the absolute value of the current scalar
        loss = torch.abs(torch.sum(predicted_energy)).item()

        logging.info("generated_params: {generated_params}")
        logging.info("hamiltonian: {hamiltonian}")
        logging.info("predicted_energy: {predicted_energy}")
        logging.info("loss: {loss}")

    return selected_type, adjs, hamiltonian, loss, generated_params
