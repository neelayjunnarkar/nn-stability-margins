#!/bin/bash
#SBATCH --job-name=rllib-train
#SBATCH --account=fc_control
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --time=24:00:00
module load python/3.7
source activate rl
python --version
python test.py
