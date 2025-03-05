#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH -p gpu
#SBATCH -o finetuning_%j.out
#SBATCH -e finetuning_%j.err

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Configure Docker to use the NVIDIA GPUs
module load nvidia/cuda/12.1
module load nvidia/cudnn/8.9.5_cuda12

# Set permissions for Hugging Face cache
mkdir -p /tmp/huggingface_cache
chmod 777 /tmp/huggingface_cache

# Build the Docker image if it doesn't exist
docker build -t llama-finetuning -f Dockerfile.finetuning .

# Run the Docker container with GPU support
docker run --gpus all \
  --shm-size=16g \
  -v $(pwd):/app \
  -v /tmp/huggingface_cache:/huggingface_cache \
  -e HF_HOME=/huggingface_cache \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  llama-finetuning

echo "End time: $(date)"