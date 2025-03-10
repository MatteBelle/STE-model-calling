#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --nodelist=faretra
#SBATCH --output=/home/belletti/sbatch_output/finetuning.out

# Invoco/eseguo script che avvia il container Docker
echo "Inizio..."
bash run_finetuning.sh
