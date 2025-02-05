#!/bin/bash
#SBATCH --job-name=GPU-bertTopic        # Job name
#SBATCH --output=log_files/bertTopic_openai.out      # Standard output and error log
#SBATCH --error=log_files/bertTopic_openai.err       # Separate file for error logs
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task (use the full node)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --partition=gpu06       # Use GPU partition
#SBATCH --qos=gpu               # Required QOS for GPU partitions
#SBATCH --time=06:00:00              # Set time limit to 8 hours (3 days)
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address

# Clear any loaded modules
module purge

# Load necessary modules
module load os/el7
module load gcc/11.2.1
module load cuda/11.7
module load python/miniforge-24.3.0  # Load conda module

# Initialize conda and activate your pre-installed environment
eval "$(conda shell.bash hook)"
conda activate bertopic_env

# Set CUDA-specific environment variables
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Optional: Print environment information (can be removed if not needed)
nvidia-smi
nvcc --version
python --version

# Change to your project directory and run your main script
cd ~/Research/narrativesBERT/
python -m BERTtopic_big_data_hpc_v7
