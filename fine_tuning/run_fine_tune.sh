#!/bin/bash
# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=metamine
#SBATCH --partition=kisski-h100         # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes to request
#SBATCH --cpus-per-task=32               # Number of CPUs to request
#SBATCH --gres=gpu:H100:4                # Request 6 H100 GPUs
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
TXT_DIR="data/txt_files"
JSON_DIR="data/json_files"
PROCESSED_DATA_DIR="processed_data"
OUTPUT_DIR="results"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"

# KEYS
WB_TOKEN="XX"
WB_PROJECT="dataset-extractor"

# Define hyperparameters
SEED=42
MAX_SEQ_LENGTH=30000     # Use full context window of the model
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
NUM_EPOCHS=3
BATCH_SIZE=1             # Keep batch size = 1
GRAD_ACCUM_STEPS=32      # Sufficient with 4 GPUs
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# Create directories
mkdir -p $PROCESSED_DATA_DIR
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Set system optimizations
# Increase shared memory size
#if [ -w /dev/shm ]; then
#  echo "Setting shared memory to maximum available"
#  df -h /dev/shm
#fi

# Setup optimized memory configurations
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
#export NCCL_SOCKET_IFNAME=^lo,docker,virbr,vmnet,vboxnet
#export NCCL_DEBUG=INFO
#export OMP_NUM_THREADS=4
#export CUDA_DEVICE_MAX_CONNECTIONS=1

export TORCH_NCCL_ENABLE_MONITORING=0  # Disable NCCL monitoring

# Ensure clean GPU state
#echo "Clearing GPU cache before starting..."
#nvidia-smi --gpu-reset

# Run with increased memory limits
#ulimit -n 65535  # Increase file descriptor limit

echo "System setup complete. Starting preprocessing..."

# Step 1: Preprocess data with full context
echo "Step 1: Preprocessing data..."
python preprocess.py \
    --txt_dir $TXT_DIR \
    --json_dir $JSON_DIR \
    --output_dir $PROCESSED_DATA_DIR \
    --model_name $MODEL_NAME \
    --seed $SEED \
    --max_seq_length $MAX_SEQ_LENGTH

# Copy DeepSpeed config
cp ds_config.json $OUTPUT_DIR/ds_config.json

# Step 2: Fine-tune the model with full context and all 6 GPUs
echo "Step 2: Fine-tuning the model..."

# Set environment variables for DeepSpeed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run fine-tuning with deepspeed
deepspeed --num_gpus=4 train.py \
    --model_name_or_path $MODEL_NAME \
    --data_dir $PROCESSED_DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $TARGET_MODULES \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --seed $SEED \
    --deepspeed $OUTPUT_DIR/ds_config.json \
    --use_flash_attention_2 \
    --enable_gradient_checkpointing \
    --report_to wandb \
    --wandb_token $WB_TOKEN \
    --wandb_project $WB_PROJECT

echo "Fine-tuning completed!"
echo "End time: $(date)"

# Copy output files to an archive directory
ARCHIVE_DIR="/mnt/vast-kisski/projects/kisski-luh-ldm-de/model_archive/dataset_extractor_$(date +%Y%m%d_%H%M%S)"
mkdir -p $ARCHIVE_DIR
echo "Archiving model and logs to $ARCHIVE_DIR"
cp -r $OUTPUT_DIR/final_model $ARCHIVE_DIR/ || echo "No final model to archive"
cp -r logs/task_${SLURM_JOB_ID}.log $ARCHIVE_DIR/ || echo "No log file to archive"