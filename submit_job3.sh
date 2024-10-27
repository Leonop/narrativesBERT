#!/bin/bash
#SBATCH --job-name=bertTopic        # Job name
#SBATCH --output=log_files/bertTopic.out      # Standard output and error log
#SBATCH --error=log_files/bertTopic.err       # Separate file for error logs
#SBATCH --partition=gpu72           # Partition for GPU nodes (qgpu)
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=16           # Number of CPU cores per task (use the full node)
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --time=72:00:00              # Set time limit to 72 hours (3 days)
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address

# First, initialize conda for bash
source /share/apps/python/miniforge-24.3.0/etc/profile.d/conda.sh

module load os/el7 gcc/11.2.1 python/miniforge-24.3.0

export PYTHONPATH="/scrfs/storage/zichengx/home/.conda/envs/bertopic_env/lib/python3.8/site-packages:$PYTHONPATH"

# 1. Activate the environment
# conda create -n bertopic_env python=3.8.19
# 2. Activate the environment
conda activate cuda_env

# # Core CUDA and ML packages
# conda install -c rapidsai -c nvidia -c conda-forge cudatoolkit=11.8.0 cudf=23.04.01 cuml=23.04.01 dask-cudf=23.04.01 cuda-python=11.8.2

# # Scientific computing packages
# conda install -c conda-forge numpy=1.23.5 pandas=1.5.3 scipy=1.10.1 numba=0.56.4

# # Install pip packages
# pip install bertopic==0.16.3 transformers==4.44.2 sentence-transformers==2.2.2 torch==2.4.1 torchvision==0.19.1 spacy==3.7.6 gensim==4.3.3 nltk==3.9.1 plotly==5.24.1

# python -m spacy download en_core_web_sm

# Install adjustText
# pip install adjustText==1.2.0

# # And also install other visualization-related dependencies that might be needed
# pip install matplotlib seaborn plotly

cd ~/Research/narrativesBERT/
# pip list | grep -E "bertopic|numpy|torch|transformers"
python BERTtopic_big_data_hpc_v3.py
