#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:2 --exclude=callisto
#SBATCH -J DAT450_Assignment_2
#SBATCH -t 00:30:00

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
alpha="$1"
beta="$2"
num_topics="$3"

out_file="alpha_${alpha}_beta_${beta}_topics_${num_topics}.txt"
exec > "$out_file" 2>&1

python3 main.py "$@"  