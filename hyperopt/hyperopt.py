import numpy 
import subprocess
import os
import sys
import json
import re
import time
os.environ["GIT_DIR"] = "/mnt/lustre/work/butz/bst080/faceGANtts/.git"

from cluster_utils import read_params_from_cmdline, save_metrics_params

def run_command(command, working_dir, log_prefix):
    print(f"[INFO] Running command: {' '.join(command)} in {working_dir}")
    stdout_log = os.path.join(working_dir, f"{log_prefix}_stdout.log")
    stderr_log = os.path.join(working_dir, f"{log_prefix}_stderr.log")

    result = subprocess.run(command, capture_output=True, text=True)

    with open(stdout_log, "w") as f:
        f.write(result.stdout)
    with open(stderr_log, "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"[ERROR] {log_prefix} failed! Check logs.")
        return None
    return result.stdout

def cluster_params_to_sacred(params):
    sacred_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                sacred_params.append(f"{key}.{sub_key}={sub_value}")
        else:
            sacred_params.append(f"{key}={value}")
    return sacred_params

def find_latest_eval_output(inference_dir, slurm_id):
    """
    Findet die neueste eval_output.txt Datei in inference_<slurm_id>/...
    """
    if not os.path.exists(inference_dir):
        raise FileNotFoundError(f"No inference directory found at {inference_dir}")

    eval_files = []
    for root, _, files in os.walk(inference_dir):
        for f in files:
            if f == "eval_output.txt":
                eval_files.append(os.path.join(root, f))

    if not eval_files:
        raise FileNotFoundError(f"No eval_output.txt found in {inference_dir}")

    eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_eval_file = eval_files[0]
    print(f"[INFO] Using latest eval_output.txt: {latest_eval_file}")

    with open(latest_eval_file, "r") as f:
        content = f.read()
        match = re.search(r"Composite Metric:\s*([\d\.\-eE]+)", content)
        if not match:
            raise ValueError(f"Could not extract Composite Metric from {latest_eval_file}")
        return float(match.group(1))

def filter_optimized_params(params):
    config_path = "hyperopt_config.json"
    if not os.path.exists(config_path):
        print("[WARNING] hyperopt_config.json not found — passing all params.")
        return params
    with open(config_path, "r") as f:
        config = json.load(f)

    allowed_keys = [p["param"] for p in config.get("optimized_params", [])]
    allowed_keys += list(config.get("fixed_params", {}).keys()) #new
    return {k: v for k, v in params.items() if k in allowed_keys}


if __name__ == "__main__":
    # 1. Read params
    params = read_params_from_cmdline()
    job_id = params.get("id", "unknown")

    # 2. Set base directory
    base_inference_dir = "/mnt/lustre/work/butz/bst080/faceGANtts"
    os.chdir(base_inference_dir)

    # 3. Set working dir
    working_dir = params.get("working_dir", base_inference_dir)
    os.makedirs(working_dir, exist_ok=True)

    # 4. Save config
    param_config_path = os.path.join(working_dir, f"params_{job_id}.json")
    with open(param_config_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"[INFO] Saved job parameters to {param_config_path}")

    # 5. Filter hyperparams
    train_eval_params = filter_optimized_params(params)

    # 6. Run training
    train_script = "/mnt/lustre/work/butz/bst080/faceGANtts/train.py"
    os.environ["HP_WORKING_DIR"] = os.path.abspath(working_dir)
    train_result = run_command(
        ["python", "-u", train_script, "main", "with"] + cluster_params_to_sacred(train_eval_params),
        working_dir,
        log_prefix=f"train_{job_id}"
    )

    # 7. Get SLURM job ID from env
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_id:
        raise RuntimeError("SLURM_JOB_ID not set in environment – cannot locate correct inference_<id> directory")

    # 8. Find latest eval result from correct subpath
    inference_base_dir = os.path.join(working_dir, f"inference_{slurm_id}")
    try:
        composite_metric = find_latest_eval_output(inference_base_dir, slurm_id)
    except Exception as e:
        raise RuntimeError(f"Fehler beim Lesen von eval_output.txt für SLURM-ID {slurm_id}: {e}")

    # 9. Report result
    metrics = {"Composite Metric": composite_metric}
    save_metrics_params(metrics, params)
