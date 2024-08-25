import os

# Set the directory path for logs
dir_path = "./log-all"

# Check if the directory already exists
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created successfully.")
else:
    print(f"Directory '{dir_path}' already exists.")

ae_types = ["GAE", "AE"]

architecture = "QAOA"
hamiltonian_models = [
    "Ising",
    "SK",
]
for ae_type in ae_types:
    for hamiltonian_model in hamiltonian_models:
        log_name = f"{architecture}_{ae_type}_{hamiltonian_model}"
        command = (
            f"nohup python -u ./main.py "
            f"--architecture {architecture} "
            f"--ae_type {ae_type} "
            f'--hamiltonian_model_types {hamiltonian_model} '
            f"> ./{dir_path}/{log_name}.log 2>&1 &"
        )
        print(command)
        os.system(command)

architecture = "HEA"
hamiltonian_models = [
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
for ae_type in ae_types:
    for hamiltonian_model in hamiltonian_models:
        log_name = f"{architecture}_{ae_type}_{hamiltonian_model}"
        command = (
            f"nohup python -u ./main.py "
            f"--architecture {architecture} "
            f"--ae_type {ae_type} "
            f'--hamiltonian_model_types {hamiltonian_model} '
            f"> ./{dir_path}/{log_name}.log 2>&1 &"
        )
        print(command)
        os.system(command)
