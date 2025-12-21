#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH -J DAT450_Assignment_2

source bravenewvenv/bin/activate

python3 main.py "$@"
