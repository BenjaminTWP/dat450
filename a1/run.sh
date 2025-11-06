#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH -J DAT450_Assignment_1
#SBATCH -t 00:30:00
#SBATCH -o prob.txt

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

python3 main.py "$@"