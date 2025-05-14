#!/bin/bash
# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=metamine            # A nice readable name of your job, to see it in the queue
#SBATCH --partition=kisski-h100         # Specify the partition name here
#SBATCH --nodes=1                     # Number of nodes to request
#SBATCH --cpus-per-task=32            # Number of CPUs to request
#SBATCH --gres=gpu:H100:1       # Request 1 A100 GPU
#SBATCH --output=logs/task_%j.log      # Standard output and error log
#SBATCH --constraint=inet
#SBATCH --time=24:00:00

# Load miniforge3 module
module load miniforge3

#module load cuda/11.8
# Initialize conda for the bash shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate your environment
ENV_PREFIX="/mnt/vast-kisski/projects/kisski-luh-ldm-de/cenv"
source activate $ENV_PREFIX


# Print environment info for debugging
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

# Run the training script
#python process_papers.py --model "meta-llama/Llama-3.2-3B-Instruct"
#python process_papers_ft.py --model "meta-llama/Llama-3.2-3B-Instruct"
python process_papers_ds.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"