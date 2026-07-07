#!/bin/bash
#SBATCH --job-name=uav_finetune
#SBATCH --partition=main-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s3335739/uav_finetune/logs/finetune_%j.log
#SBATCH --error=/home/s3335739/uav_finetune/logs/finetune_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load python/3.10.7
source /home/s3335739/uav_env/bin/activate

export LD_LIBRARY_PATH=/home/s3335739/uav_finetune/libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python3 /home/s3335739/uav_finetune/scripts/finetune.py

echo "Job finished: $(date)"
