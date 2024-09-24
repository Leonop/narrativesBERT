#!/bin/bash
#SBATCH --job-name=bertTopic        # Job name
#SBATCH --output=my_gpu_job01.out      # Standard output and error log
#SBATCH --error=my_gpu_job01.err       # Separate file for error logs
#SBATCH --partition=qgpu72           # Partition for GPU nodes (qgpu)
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=12           # Number of CPU cores per task (use the full node)
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --time=72:00:00              # Set time limit to 72 hours (3 days)
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address

# Initialize conda for this session
source ~/miniconda3/etc/profile.d/conda.sh

# Load Conda module
module load cuda/11.2
#module load python/3.8     # Correct Python module
#module load anaconda/2021.05

# Create a new Conda environment
#conda create --name cuda_env python=3.8

# Change to the directory where you submitted the job
cd ~/home/zichengx/Research/narrativesBERT/
conda activate cuda_env

# Check GPU availability
nvidia-smi

# Activate the virtual environment
#source /scrfs/storage/zichengx/home/Research/narrativesBERT/cuda_env/bin/activate
#conda activate /home/zichengx/Research/narrativesBERT/cuda_env/bin/activate

# Install necessary packages (only if they are not already installed)
pip install  -r requirements.txt
# conda install -c rapidsai -c nvidia -c conda-forge  cuml=23.4.1 python=3.8.19 cudatoolkit=11.2
# Instead of running Jupyter, run your Python script directly (convert .ipynb to .py first)
pip uninstall cupy
pip install cupy-cuda112

python BERTtopic_big_data_hpc.py

# Optional: If you need to do any additional work, add your commands here
