#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:2 --exclude=callisto
#SBATCH -J DAT450_Assignment_2

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

python3 main.py "$@"
