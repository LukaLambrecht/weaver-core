#!/bin/bash
#SBATCH --job-name=weaver_test
#SBATCH --output=logs/weaver_test_%j.out
#SBATCH --error=logs/weaver_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Change to the weaver-core directory
cd /users/tgillin/files/weaver-core/weaver

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

python evaluation/evaluate.py -i /users/tgillin/files/weaver-core/weaver/output_test/output.root

