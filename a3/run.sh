#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1 --exclude=callisto
#SBATCH -J DAT450_Assignment_3
#SBATCH -t 00:30:00
#SBATCH -o a3_out.txt

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

python3 test.py  