#!/bin/bash
# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=metamine
#SBATCH --partition=kisski-h100         # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes to request
#SBATCH --cpus-per-task=32               # Number of CPUs to request
#SBATCH --gres=gpu:H100:1                # Request 6 H100 GPUs
#SBATCH --output=logs/task_%j.log        # Standard output and error log
#SBATCH --constraint=inet
#SBATCH --time=24:00:00                  # Extended time for training


# Load miniforge3 module
module load miniforge3

# Initialize conda for the bash shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate your environment
ENV_PREFIX="/mnt/vast-kisski/projects/kisski-luh-ldm-de/cenv"
source activate $ENV_PREFIX

# Print environment info for debugging
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Define paths
CHECK_DIR="results/checkpoint-444"
OUTPUT_DIR="results/final_model"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"



# Step 1: Preprocess data with full context
python recover_model.py \
    --base_model $MODEL_NAME \
    --checkpoint_dir $CHECK_DIR \
    --output_dir $OUTPUT_DIR


echo "Done Saving Model"
