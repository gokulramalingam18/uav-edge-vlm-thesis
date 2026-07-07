#!/bin/bash
#SBATCH --job-name=quant_imatrix
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --output=/home/s3335739/uav_finetune/logs/quant_imatrix_%j.log
#SBATCH --error=/home/s3335739/uav_finetune/logs/quant_imatrix_%j.err

export LD_LIBRARY_PATH=/home/s3335739/uav_finetune/libs:$LD_LIBRARY_PATH
module load python/3.10.7 2>/dev/null
source /home/s3335739/uav_env/bin/activate
echo "Job started: $(date)"
echo "Node: $(hostname)"

/home/s3335739/uav_finetune/llama.cpp/build/bin/llama-quantize \
    --imatrix /home/s3335739/uav_finetune/output/imatrix.dat \
    /home/s3335739/uav_finetune/output/qwen_uav_text.gguf \
    /home/s3335739/uav_finetune/output/qwen_uav_text_q4_imatrix.gguf \
    Q4_K_M 2>&1

echo "Job finished: $(date)"
