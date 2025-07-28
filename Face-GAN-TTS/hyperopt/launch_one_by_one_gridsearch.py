import json
import os
import subprocess
from copy import deepcopy
from time import sleep

BASE_CONFIG_PATH = "original_config.json"
OUTPUT_DIR = "one_by_one_configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(BASE_CONFIG_PATH, "r") as f:
    base_config = json.load(f)

original_hyperparams = base_config["hyperparam_list"]
base_config.pop("hyperparam_list", None)

for param_block in original_hyperparams:
    param_name = param_block["param"]
    values = param_block["values"]

    for val in values:
        new_config = deepcopy(base_config)
        new_config["hyperparam_list"] = [{
            "param": param_name,
            "values": [val]
        }]

        unique_name = f"onebyone_{param_name.replace('.', '_')}={str(val).replace('.', '_')}"
        new_config["optimization_procedure_name"] = unique_name

        config_filename = f"{unique_name}.json"
        config_path = os.path.join(OUTPUT_DIR, config_filename)
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=2)

        print(f"[INFO] Config saved: {config_path}")
        print(f"[INFO] Launching in tmux: {unique_name}")

        # vor dem Launch: Pfad berechnen
        working_dir = os.path.abspath(f"/mnt/lustre/work/butz/bst080/faceGANtts/hyperopt/onebyone_runs/{unique_name}")
        os.makedirs(working_dir, exist_ok=True)


        #tmux_cmd = f'tmux new-session -d -s {unique_name} "python3 -m cluster_utils.grid_search {config_path}"'
        tmux_cmd = (
            f'tmux new-session -d -s {unique_name} "'
            f'export HP_WORKING_DIR={working_dir} && '
            f'python3 -m cluster_utils.grid_search {config_path}"'
        )
        subprocess.run(tmux_cmd, shell=True)

        # Optional kleine Pause
        sleep(1)
