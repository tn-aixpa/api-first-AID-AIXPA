from huggingface_hub import hf_hub_download
import os

# Set model repository and filename
repo_id = "Stefano-M/aixpa_amicifamiglia_short_prompt"  # Example adapter repo
file_name = "adapter_model.safetensors"  # Example file to download
local_folder = "/adapters/aixpa"

# Ensure local directory exists
os.makedirs(local_folder, exist_ok=True)

# Download adapter
local_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_folder)

file_name = "adapter_config.json"  # Example file to download
local_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_folder)
file_name = "README.md"
local_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_folder)


print(f"Adapter downloaded to: {local_path}")