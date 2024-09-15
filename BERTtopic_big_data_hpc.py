# Using HPC to run BERTopic on big data with GPU
# python version : 3.8.5
# Author: Zicheng Xiao
# Date: 2024-09-11
# Summary of code structure:
# 1. Install the required packages
# 2. Load the data
# 3. Run BERTopic on the data
# 4. Save the results
# 5. Visualize the results
# 6. Save the visualization
# 7. Save the model
# 8. Save the topics
# 9. Save the topic probabilities
# 10. Save the topic words
# 11. Save the topic frequencies
# 12. Save the topic sizes
# 13. Save the topic embeddings
# 14. Save the topic hierarchy
# 15. Save the topic coherence
# 16. Save the topic count


import subprocess
import sys
import pandas as pd
import numpy as np
from bertopic import BERTopic

folder_path = '/scrfs/storage/zichengx/home/Research/narrativesBERT'
data_path = '/scrfs/storage/zichengx/home/Research/narrativesBERT/data/earnings_calls_20231017.csv'

# BERTopic
# Enableing the GPU for BERTopic
# First, you'll need to enable GPUs for the notebook:
# Navigate to Edit 🡒 Notebook Setting
# Select GPU from the Hardware Accelerator dropdown

def install_packages():
    # List of packages to install
    packages = [
        "git+https://github.com/MaartenGr/BERTopic.git@master",
        "cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com",
        "cuml-cu12 --extra-index-url=https://pypi.nvidia.com",
        "cugraph-cu12 --extra-index-url=https://pypi.nvidia.com",
        "cupy-cuda12x -f https://pip.cupy.dev/aarch64",
        "safetensors",
        "datasets",
        "datashader",
        "adjustText",
        "sentence-transformers==2.2.2"
        "bertopic",
    ]

    # Install each package
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_data():
    data = pd.read_csv(data_path)
    return data

if __name__ == "__main__":
    install_packages()