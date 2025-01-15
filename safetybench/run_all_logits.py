import os
import json
import subprocess
import fire
from tqdm import tqdm

def read_configs(configs_path):
    with open(configs_path, 'r') as f:
        configs = json.load(f)

    return configs

def run(
    script_name: str,
    configs_path: str,
    datapath: str,
    cuda_visible_devices: int = 1,
    gpu_memory_utilization: float = 0.9
):
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    
    configs = read_configs(configs_path)

    successful_configs = []
    failed_configs = []
    
    for config in tqdm(configs):
        model_name = config.get("model_name")
        savepath = config.get("savepath")
        tokenizer_name = config.get("tokenizer_name")
        model_dtype = config.get("model_dtype")

        print('\n\n\n' + '='*10 + f' MODEL: {model_name} ' + '='*10 + '\n\n\n')
        print(f'params: {config}')

        command = [
            "python3", script_name,
            "--datapath", datapath,
            "--savepath", savepath,
            "--model_name", model_name,
            "--gpu_memory_utilization", str(gpu_memory_utilization)
        ]
        if tokenizer_name:
            command.append("--tokenizer_name")
            command.append(tokenizer_name)
        if model_dtype:
            command.append("--model_dtype")
            command.append(model_dtype)

        str_command = " ".join(command)
        print(f'\nRunning command: {str_command}')

        try:
            subprocess.run(command, env=env, check=True)
            # result = subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            successful_configs.append(config)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            failed_configs.append({"config": config, "error": e.stderr})

    print('\n\n\n' + '='*10 + f' EXECUTION SUMMARY ' + '='*10 + '\n')
    print(f'success: {len(successful_configs)}')
    print(f' failed: {len(failed_configs)}')
    print(f"\nSuccessful:")
    for config in successful_configs:
        print(f"- {config['model_name']}")

    print(f"\nFailed:")
    for failure in failed_configs:
        print(f"- {failure['config']['model_name']}")
        # print(f"  Error: {failure['error']}")


if __name__ == "__main__":
    fire.Fire(run)