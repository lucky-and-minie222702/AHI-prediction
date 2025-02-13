#!/bin/bash
#SBATCH --job-name=VY_TRIET_MODEL        # Job name
#SBATCH --output=output.txt        # Output file
#SBATCH --error=error.txt          # Error log file
#SBATCH --time=51:30:00            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu            # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=96G                   # Memory per node
#SBATCH --gres=gpu:gu01:1   

module load cuda/12.3
module load python/3.10 

python3.10 src/model_ecg_ah.py