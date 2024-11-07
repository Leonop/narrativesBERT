#!/bin/bash
#SBATCH --job-name=bertTopic        # Job name
#SBATCH --output=log_files/bertTopic.out      # Standard output and error log
#SBATCH --error=log_files/bertTopic.err       # Separate file for error logs
#SBATCH --partition=gpu72           # Partition for GPU nodes (qgpu)
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=32           # Number of CPU cores per task (use the full node)
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --time=72:00:00              # Set time limit to 72 hours (3 days)
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address

# 1. Load the required modules
module load os/el7 gcc/11.2.1 python/miniforge-24.3.0

# 2. Activate the environment
conda activate bertopic_env

# 3. Install the required packages
# Install packages in specific order with compatible versions
# pip install torch==2.4.1 torchvision==0.19.1
# pip install tokenizers==0.19.0
# pip install transformers==4.44.2
# pip install sentence-transformers==2.2.2
# pip install huggingface-hub==0.23.2
# pip install bertopic==0.16.3 gensim==4.3.3 nltk==3.9.1 plotly==5.24.1
# pip install joblib==1.2.0
# module load cuda/11.8
# conda install -c rapidsai -c nvidia -c conda-forge cudatoolkit=11.8.0 cudf=23.04.01 cuml=23.04.01 dask-cudf=23.04.01 cuda-python=11.8.2
# conda install -c conda-forge numpy=1.23.5 pandas=1.5.3 scipy=1.10.1 numba=0.56.4

# python -m spacy download en_core_web_sm
# pip install -U kaleido

cd ~/Research/narrativesBERT/
python BERTtopic_big_data_hpc_v3.py
