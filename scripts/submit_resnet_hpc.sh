#!/bin/bash
# SLURM submission script for ResNet20 CIFAR-10 iterative pruning
# Modify SLURM directives as needed for your HPC cluster

#SBATCH --job-name=resnet20-cifar10
#SBATCH --output=logs/resnet20-%j.out
#SBATCH --error=logs/resnet20-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Set environment
module load python/3.11  # Adjust to your cluster's Python module
module load cuda/12.1    # Adjust to your CUDA version

# Activate virtual environment (if using one)
source /path/to/venv/bin/activate  # Adjust to your virtual env path

# Navigate to project directory
cd /home/nr_havv/nr_haml2025/endomet/irwl1-adaptation  # Adjust to your project path

# Create logs directory
mkdir -p logs

# Run the training script with HPC-optimized settings
python scripts/resnet_train.py

echo "Job completed at $(date)"
