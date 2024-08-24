import logging
from pathlib import Path

from models.GAE import GAE
from trainer import (
    compare_generated_and_trained_qnn,
    train_ae,
    train_gae,
    train_hypernet,
)
from utils.utils import (
    load_model,
    main_parser,
    create_directory,
    save_args_to_json,
    set_random_seed,
    set_log,
    generate_hidden_dims,
)
from utils.global_variable import *


if __name__ == "__main__":
    # 设置随机种子
    set_random_seed(seed)

    # TODO CHECK before running
    config_file = "./config/test-qaoa.json"

    args = main_parser(config_file)

    # TODO CHECK for debug
    # args.architecture = "QAOA"
    args.ae_type = "GAE"
    # args.hamiltonian_model_types = ["Ising"]
    args.hamiltonian_model_types = ["SK"]
    # args.hamiltonian_model_types = ["HM-AILO"]
    # args.hamiltonian_model_type_weights = None
    # args.test_hamiltonian_model_types = ["Ising"]
    # args.test_hamiltonian_model_type_weights = None
    args.gae_n_epochs = 30
    # args.gae_n_batches = 1
    # args.hypernet_n_epochs = 2
    # args.hypernet_n_batches = 2
    # args.n_random_hamiltonian_tobe_train = 2
    # args.n_encoder_output = 20
    # args.n_train_from_scratch_steps = 10

    args.encoder_hidden_dims = generate_hidden_dims(
        3 * (args.n_qubits**2), min_dim=args.n_encoder_output
    )
    args.decoder_hidden_dims = args.encoder_hidden_dims[::-1]

    # 创建实验名称
    args.exp_name = f"{args.architecture}_qubits{args.n_qubits}_layers{args.n_layers}_hamiltonian{args.hamiltonian_model_types}_tf{args.has_transverse_field}_{args.ae_type}"
    result_path = Path("results")
    args.exp_path = result_path / args.exp_name
    create_directory(args.exp_path)
    create_directory(args.exp_path / random_trained_result_dir)

    save_args_to_json(args, args.exp_path)
    set_log(args.exp_path)

    # 打印所有参数
    logging.info("Parsed Arguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    if args.ae_type == "GAE":
        ae_model, losses = train_gae(
            args.n_qubits,
            args.encoder_hidden_dims,
            args.n_encoder_output,
            args.gae_n_epochs,
            args.gae_n_batches,
            args.gae_batch_size,
            args.hamiltonian_model_types,
            args.hamiltonian_model_type_weights,
            args.has_transverse_field,
            args.exp_path,
        )
    elif args.ae_type == "AE":
        ae_model, losses = train_ae(
            args.n_qubits,
            args.encoder_hidden_dims,
            args.decoder_hidden_dims,
            args.gae_n_epochs,
            args.gae_n_batches,
            args.gae_batch_size,
            args.hamiltonian_model_types,
            args.hamiltonian_model_type_weights,
            args.has_transverse_field,
            args.exp_path,
        )
    else:
        raise ValueError("Invalid AE type!")

    # # Load the pre-trained GAE model
    # gae_model_path = args.exp_path / "gae_model.pth"  # Path to the saved GAE model
    # gae = load_model(GAE, gae_model_path, args.n_qubits, args.encoder_hidden_dims, args.n_encoder_output)
    # gae_encoder = gae.encoder

    train_hypernet(
        args.architecture,
        args.n_qubits,
        args.n_layers,
        args.ae_type,
        ae_model.encoder,
        args.hypernet_n_epochs,
        args.hypernet_n_batches,
        args.hypernet_batch_size,
        args.hamiltonian_model_types,
        args.hamiltonian_model_type_weights,
        args.has_transverse_field,
        args.exp_path,
    )

    compare_generated_and_trained_qnn(
        args.ae_type,
        args.exp_name,
        n_steps=args.n_train_from_scratch_steps,
        n_random_architectures=args.n_random_hamiltonian_tobe_train,
    )
