import logging
import json5
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pathlib import Path

from models.AE import Autoencoder
from models.GAE import GAE
from models.Hypernetwork import Hypernetwork
from models.HEA import HEA
from models.QAOA import QAOA
from utils.data_generator import generate_random_data
from utils.utils import (
    load_model,
    load_args_from_json,
    generate_and_evaluate_hea,
    generate_and_evaluate_qaoa,
)
from utils.plot_utils import plot_loss_with_error_bars, plot_losses
from utils.global_variable import *


def train_ae(
    n_qubits,
    encoder_hidden_dims,
    decoder_hidden_dims,
    n_epochs,
    n_batches,
    batch_size,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
    exp_path,
):
    input_dim = int((3 + 3 / 2) * n_qubits * (n_qubits - 1) + 1)

    ae = Autoencoder(input_dim, encoder_hidden_dims, decoder_hidden_dims)
    optimizer = optim.Adam(ae.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    ae.train()
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            adjs_flattened_batch = []

            # Generate a batch of data
            for _ in range(batch_size):
                # TODO 是否只用当前类型的哈密顿量训练AE？
                selected_type, x, adjs, hamiltonian, adjs_flattened = (
                    generate_random_data(
                        n_qubits,
                        hamiltonian_model_types=hamiltonian_model_types,
                        hamiltonian_model_type_weights=hamiltonian_model_type_weights,
                        has_transverse_field=has_transverse_field,
                    )
                )
                adjs_flattened_batch.append(adjs_flattened)

            adjs_flattened_batch = torch.stack(adjs_flattened_batch)

            optimizer.zero_grad()
            # Forward pass
            _, adj_reconstructed = ae(adjs_flattened_batch)

            # Compute loss
            loss = 0
            for i in range(batch_size):
                loss += loss_fn(adj_reconstructed[i], adjs_flattened_batch[i])
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        # # 使用 logging 打印第一层的权重
        # logging.info(f"First layer weights: {ae.encoder.encoder[0].weight}")
        # # 如果第一层有偏置项，也可以打印
        # logging.info(f"First layer bias: {ae.encoder.encoder[0].bias}")

    # Save the model if needed
    torch.save(ae.state_dict(), exp_path / "ae_model.pth")

    # Save the losses to a JSON file
    ae_loss_file = exp_path / "ae_losses.json"
    with open(ae_loss_file, "w") as f:
        json5.dump(losses, f)

    ae_loss_fig_file = exp_path / "fig_ae_losses.png"
    plot_losses(ae_loss_file, fig_path=ae_loss_fig_file)

    return ae, losses


# Define the training function
def train_gae(
    n_qubits,
    encoder_hidden_dims,
    n_encoder_output,
    n_epochs,
    n_batches,
    batch_size,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
    exp_path,
):
    gae = GAE(n_qubits, n_qubits, encoder_hidden_dims, n_encoder_output)
    optimizer = optim.Adam(gae.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    gae.train()
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            x_batch, adjs_batch, hamiltonian_batch = [], [], []

            # Generate a batch of data
            for _ in range(batch_size):
                selected_type, x, adjs, hamiltonian, adjs_flattened = (
                    generate_random_data(
                        n_qubits,
                        hamiltonian_model_types=hamiltonian_model_types,
                        has_transverse_field=has_transverse_field,
                    )
                )
                x_batch.append(x)
                adjs_batch.append(adjs)
                hamiltonian_batch.append(hamiltonian)

            x_batch = torch.stack(x_batch)
            # adjs_batch = [
            #     torch.stack([adjs[i] for adjs in adjs_batch])
            #     for i in range(num_relations)
            # ]

            optimizer.zero_grad()
            # Forward pass
            _, adj_reconstructed = gae(x_batch, adjs_batch)

            # Compute loss
            loss = 0
            for i in range(batch_size):
                loss += loss_fn(adj_reconstructed[i], adjs_batch[i])
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Save the model if needed
    torch.save(gae.state_dict(), exp_path / "ae_model.pth")

    # Save the losses to a JSON file
    ae_loss_file = exp_path / "ae_losses.json"
    with open(ae_loss_file, "w") as f:
        json5.dump(losses, f)

    ae_loss_fig_file = exp_path / "fig_ae_losses.png"
    plot_losses(ae_loss_file, fig_path=ae_loss_fig_file)

    return gae, losses


def train_hypernet(
    architecture,
    n_qubits,
    n_layers,
    ae_encoder_type,
    ae_encoder,
    hypernet_n_epochs,
    hypernet_n_batches,
    hypernet_batch_size,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
    exp_path,
):
    # # Load the pre-trained GAE model
    # ae_model_path = exp_path / "ae_model.pth"  # Path to the saved GAE model
    # gae = load_model(GAE, ae_model_path, n_qubits, encoder_hidden_dims, n_encoder_output)
    # ae_encoder = gae.encoder

    if architecture == "HEA":
        train_hypernet_HEA(
            n_qubits,
            n_layers,
            ae_encoder_type,
            ae_encoder,
            hypernet_n_epochs,
            hypernet_n_batches,
            hypernet_batch_size,
            hamiltonian_model_types,
            hamiltonian_model_type_weights,
            has_transverse_field,
            exp_path,
        )
    elif architecture == "QAOA":
        # assert hamiltonian_model_types == [
        #     "Ising"
        # ], "hamiltonian_model_type for QAOA only support Ising for now"
        # 定义 Ising SK 支持的类型
        ising_sk_types = ["Ising", "SK"]  # 根据你的需求列出所有可能的类型
        # 断言 hamiltonian_model_types 是否属于 ising_sk_types 之一
        assert all(
            model_type in ising_sk_types for model_type in hamiltonian_model_types
        ), "hamiltonian_model_type for QAOA only supports types within Ising or SK"

        train_hypernet_QAOA(
            n_qubits,
            n_layers,
            ae_encoder_type,
            ae_encoder,
            hypernet_n_epochs,
            hypernet_n_batches,
            hypernet_batch_size,
            hamiltonian_model_types,
            hamiltonian_model_type_weights,
            has_transverse_field,
            exp_path,
        )
    else:
        raise ValueError("Invalid architecture choice.")


# Function to train the Hypernetwork
def train_hypernet_HEA(
    n_qubits,
    n_layers,
    ae_encoder_type,
    ae_encoder,
    n_epochs,
    n_batches,
    batch_size,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
    exp_path,
):
    n_output_dim_each_layer = n_qubits * 3
    hypernet = Hypernetwork(
        n_qubits, n_layers, ae_encoder.n_encoder_output, n_output_dim_each_layer
    )
    qnn = HEA(n_qubits, n_layers)

    optimizer = optim.Adam(
        list(hypernet.parameters()) + list(qnn.parameters()), lr=0.001
    )

    hypernet.train()
    qnn.train()
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            x_batch, adjs_batch, hamiltonian_batch, adjs_flattened_batch = (
                [],
                [],
                [],
                [],
            )

            # Generate a batch of data
            for _ in range(batch_size):
                selected_type, x, adjs, hamiltonian, adjs_flattened = (
                    generate_random_data(
                        n_qubits,
                        hamiltonian_model_types=hamiltonian_model_types,
                        hamiltonian_model_type_weights=hamiltonian_model_type_weights,
                        has_transverse_field=has_transverse_field,
                    )
                )
                x_batch.append(x)
                adjs_batch.append(adjs)
                hamiltonian_batch.append(hamiltonian)
                adjs_flattened_batch.append(adjs_flattened)

            optimizer.zero_grad()
            # Use GAE encoder to generate latent space representations, disable gradient computation
            with torch.no_grad():
                if ae_encoder_type == "AE":
                    adjs_flattened_batch = torch.stack(adjs_flattened_batch)
                    z_batch = ae_encoder(
                        adjs_flattened_batch,
                    )
                elif ae_encoder_type == "GAE":
                    z_batch = ae_encoder(x_batch, adjs_batch)
                else:
                    raise ValueError("Invalid AE encoder type.")

            # Hypernetwork generates QNN parameters
            paramsz_batch = hypernet(z_batch)
            paramsz_batch = paramsz_batch.view(-1, n_layers, qnn.n_qubits, 3)
            loss = 0
            # Forward propagate through QNN, compute ground state energy
            predicted_energy = qnn(paramsz_batch, hamiltonian_batch)

            # Compute loss, the lower the ground state energy the better
            # loss = torch.sum(torch.abs(predicted_energy))
            loss = torch.norm(predicted_energy, p=2)  # L2范数
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        logging.info(
            f"train_hypernet_HEA Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}"
        )

    # Save the model if needed
    torch.save(hypernet.state_dict(), exp_path / "hypernet.pth")

    # Save the losses to a JSON file
    hypernet_loss_file = exp_path / "hypernet_losses.json"
    with open(hypernet_loss_file, "w") as f:
        json5.dump(losses, f)

    hypernet_loss_fig_file = exp_path / "fig_hypernet_losses.png"
    plot_losses(hypernet_loss_file, fig_path=hypernet_loss_fig_file)

    return hypernet, losses


def train_hypernet_QAOA(
    n_qubits,
    n_layers,
    ae_encoder_type,
    ae_encoder,
    n_epochs,
    n_batches,
    batch_size,
    hamiltonian_model_types,
    hamiltonian_model_type_weights,
    has_transverse_field,
    exp_path,
):
    n_output_dim_each_layer = 2
    hypernet = Hypernetwork(
        n_qubits, n_layers, ae_encoder.n_encoder_output, n_output_dim_each_layer
    )

    optimizer = optim.Adam(list(hypernet.parameters()), lr=0.001)

    hypernet.train()
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            x_batch, adjs_batch, hamiltonian_batch, adjs_flattened_batch = (
                [],
                [],
                [],
                [],
            )

            # Generate a batch of data
            for _ in range(batch_size):
                selected_type, x, adjs, hamiltonian, adjs_flattened = (
                    generate_random_data(
                        n_qubits,
                        hamiltonian_model_types=hamiltonian_model_types,
                        hamiltonian_model_type_weights=hamiltonian_model_type_weights,
                        has_transverse_field=has_transverse_field,
                    )
                )
                x_batch.append(x)
                adjs_batch.append(adjs)
                hamiltonian_batch.append(hamiltonian)
                adjs_flattened_batch.append(adjs_flattened)

            optimizer.zero_grad()
            # Use GAE encoder to generate latent space representations, disable gradient computation
            with torch.no_grad():
                if ae_encoder_type == "AE":
                    adjs_flattened_batch = torch.stack(adjs_flattened_batch)
                    z_batch = ae_encoder(
                        adjs_flattened_batch,
                    )
                elif ae_encoder_type == "GAE":
                    z_batch = ae_encoder(x_batch, adjs_batch)
                else:
                    raise ValueError("Invalid AE encoder type.")

            # Hypernetwork generates QNN parameters
            paramsz_batch = hypernet(z_batch)
            paramsz_batch = paramsz_batch.view(-1, 2, n_layers)
            loss = 0
            for i, hamiltonian in enumerate(hamiltonian_batch):
                qnn = QAOA(n_qubits, n_layers, hamiltonian)
                qnn.train()
                # 前向传播QNN，计算基态能量
                predicted_energy = qnn(paramsz_batch[i][0], paramsz_batch[i][1])

                # 计算损失，基态能量越小越好
                loss += torch.sum(torch.abs(predicted_energy))

            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Save the model if needed
    torch.save(hypernet.state_dict(), exp_path / "hypernet.pth")

    # Save the losses to a JSON file
    hypernet_loss_file = exp_path / "hypernet_losses.json"
    with open(hypernet_loss_file, "w") as f:
        json5.dump(losses, f)

    hypernet_loss_fig_file = exp_path / "fig_hypernet_losses.png"
    plot_losses(hypernet_loss_file, fig_path=hypernet_loss_fig_file)

    return hypernet, losses


# Function to train the HEA directly with random initialization
def train_hea(
    n_qubits,
    n_layers,
    n_steps,
    hamiltonian,
    result_prefix,
    init_params=None,
):
    # Initialize the HEA model
    hea = HEA(n_qubits, n_layers)
    losses = []

    hamiltonian_batch = []
    hamiltonian_batch.append(hamiltonian)

    if init_params is not None:
        # 复制一份新的张量，并确保 requires_grad=True
        params_batch = init_params.clone().detach().requires_grad_(True)
    else:
        # Randomly initialize HEA parameters
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        params_batch = torch.rand((1, *shape), requires_grad=True)

    # Define optimizer
    optimizer = optim.Adam([params_batch], lr=0.001)

    # Training loop
    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward propagate through HEA, compute ground state energy
        predicted_energy = hea(params_batch, hamiltonian_batch)

        # Compute loss, the absolute value of the current scalar
        loss = torch.abs(torch.sum(predicted_energy))
        losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        logging.info(f"Step {step+1}/{n_steps}, Loss: {loss:.4f}")

        if step == 0:
            logging.info("params_batch: {params_batch}")
            logging.info("hamiltonian_batch: {hamiltonian_batch}")
            logging.info("predicted_energy: {predicted_energy}")
            logging.info("loss: {loss}")

    # Save the model if needed
    torch.save(params_batch, f"{result_prefix}_model.pth")

    # Save the losses to a JSON file
    with open(
        f"{result_prefix}_losses.json",
        "w",
    ) as f:
        json5.dump(losses, f)

    return params_batch, losses


# Function to train QAOA model directly with random initialization
def train_qaoa(
    n_qubits,
    n_layers,
    n_steps,
    hamiltonian,
    result_prefix,
    init_params=None,
):
    # Initialize the QAOA model
    qaoa = QAOA(n_qubits, n_layers, hamiltonian)
    losses = []

    if init_params is not None:
        # 复制一份新的张量，并确保 requires_grad=True
        params_gamma = init_params[0].clone().detach().requires_grad_(True)
        params_alpha = init_params[1].clone().detach().requires_grad_(True)
    else:
        params_gamma = torch.rand(n_layers, requires_grad=True)
        params_alpha = torch.rand(n_layers, requires_grad=True)

    # Define optimizer
    optimizer = optim.Adam([params_gamma, params_alpha], lr=0.001)

    # Training loop
    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward propagate through QAOA, compute ground state energy
        predicted_energy = qaoa(params_gamma, params_alpha)

        # Compute loss, the absolute value of the current scalar
        loss = torch.abs(torch.sum(predicted_energy))
        losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        logging.info(f"Step {step+1}/{n_steps}, Loss: {loss:.4f}")

    # Save the parameters and losses
    torch.save(
        {"params_gamma": params_gamma, "params_alpha": params_alpha},
        f"{result_prefix}_model.pth",
    )
    # Save the losses to a JSON file
    with open(
        f"{result_prefix}_losses.json",
        "w",
    ) as f:
        json5.dump(losses, f)

    return {"params_gamma": params_gamma, "params_alpha": params_alpha}, losses


def compare_generated_and_trained_qnn(
    ae_type, exp_name, n_steps=100, n_random_architectures=5
):
    result_path = Path(f"./results/{exp_name}")

    # Set model paths and parameters
    ae_model_path = result_path / "ae_model.pth"  # Path to the saved GAE model
    hypernet_model_path = (
        result_path / "hypernet.pth"
    )  # Path to the saved Hypernetwork model
    json_file_path = result_path / "args.json"

    # Load the args from JSON
    args = load_args_from_json(json_file_path)

    # Load pre-trained models
    if ae_type == "AE":
        input_dim = int((3 + 3 / 2) * args.n_qubits * (args.n_qubits - 1) + 1)
        ae_model = load_model(
            Autoencoder,
            ae_model_path,
            input_dim,
            args.encoder_hidden_dims,
            args.decoder_hidden_dims,
        )
    elif ae_type == "GAE":
        ae_model = load_model(
            GAE,
            ae_model_path,
            args.n_qubits,
            args.n_qubits,
            args.encoder_hidden_dims,
            args.n_encoder_output,
        )
    else:
        raise ValueError("Invalid as_type. Choose either AE or GAE.")

    ae_encoder = ae_model.encoder

    if args.architecture == "HEA":
        n_output_dim_each_layer = args.n_qubits * 3
        generate_and_evaluate_func = generate_and_evaluate_hea
        train_func = train_hea
    elif args.architecture == "QAOA":
        n_output_dim_each_layer = 2
        generate_and_evaluate_func = generate_and_evaluate_qaoa
        train_func = train_qaoa

    all_random_losses = []
    final_losses = []

    for architectures_seed in range(n_random_architectures):
        # Load pre-trained models
        hypernet = load_model(
            Hypernetwork,
            hypernet_model_path,
            args.n_qubits,
            args.n_layers,
            args.n_encoder_output,
            n_output_dim_each_layer,
        )

        # Generate and evaluate the HEA model
        selected_type, adjs, hamiltonian, final_loss, generated_params = (
            generate_and_evaluate_func(
                args.n_qubits,
                args.n_layers,
                ae_type,
                ae_encoder,
                hypernet,
                args.test_hamiltonian_model_types,
                args.test_hamiltonian_model_type_weights,
                args.has_transverse_field,
            )
        )
        final_losses.append(final_loss)
        logging.info(f"Selected type: {selected_type}")
        logging.info(f"Hamiltonian: {hamiltonian}")
        logging.info(f"adjs: {adjs}")
        logging.info(f"Final Loss for QNN: {final_loss:.4f}")

        result_prefix = f"{args.exp_path}/{random_trained_result_dir}/{architectures_seed}_{selected_type}_generated"
        params_batch, generated_train_losses = train_func(
            args.n_qubits,
            args.n_layers,
            n_steps,
            hamiltonian,
            result_prefix,
            init_params=generated_params,
        )
        plot_losses(
            f"{result_prefix}_losses.json",
            horizontal_line=final_loss,
            fig_path=f"{result_prefix}_losses.png",
        )

        result_prefix = f"{args.exp_path}/{random_trained_result_dir}/{architectures_seed}_{selected_type}_random"
        params_batch, random_train_losses = train_func(
            args.n_qubits,
            args.n_layers,
            n_steps,
            hamiltonian,
            result_prefix,
            init_params=None,
        )
        all_random_losses.append(random_train_losses)
        plot_losses(
            f"{result_prefix}_losses.json",
            horizontal_line=final_loss,
            fig_path=f"{result_prefix}_losses.png",
        )

    # # Calculate mean final loss
    # mean_final_loss = np.mean(final_losses)

    # # Plot losses with error bars
    # fig_path = args.exp_path / random_trained_result_dir / "loss_with_error_bars.png"
    # plot_loss_with_error_bars(all_random_losses, mean_final_loss, fig_path)
