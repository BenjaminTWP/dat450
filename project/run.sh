#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH -J MT
#SBATCH --output=runs/%x_jobId_%j.out
#SBATCH --error=runs/%x_jobId_%j.err

source bravenewvenv/bin/activate

python3 main.py "$@"
