import os
from typing import Dict, List
import pandas as pd
# MIN_CLUSTER_SIZE = 20
# N_TOPICS = 100
# N_TOP_WORDS = 15

# Directory locations

PROJECT_DIR = os.getcwd()
data_folder = os.path.join(PROJECT_DIR, "data")
model_folder = os.path.join(PROJECT_DIR, "model")
output_folder = os.path.join(PROJECT_DIR, "output")
output_fig_folder = os.path.join(output_folder, "fig")
data_filename = 'earnings_calls_20231017.csv'
stop_list = pd.read_csv(os.path.join(data_folder, "stoplist.csv"))['stopwords'].tolist()
MODEL_SCORES = os.path.join(output_folder, "model_scores.txt")
DATE_COLUMN = "transcriptcreationdate_utc"
TOPIC_SCATTER_PLOT = os.path.join(output_fig_folder, "topic_scatter_plot.pdf")
num_topic_to_plot = 20 # top_N topics to plot

TEXT_COLUMN = "componenttext" # the column in the main earnings call data that contains the earnings transcript
NROWS = 10000 # number of rows to read from the csv file
CHUNK_SIZE = 1000 # number of rows to read at a time
YEAR_FILTER = 2025 # filter the data based on the year
# Batch Size for Bert Topic Model Training in BERTopic_big_data_hpc.py
BATCH_SIZE = 1000

# create a list of parameters to search over using GridSearchCV
N_NEIGHBORS = [10] 
N_COMPONENTS = [5]
MIN_DIST = [0.5]
MIN_SAMPLES = [5]
MIN_CLUSTER_SIZE = [100] # Number of Topics in Topic Model
N_TOPICS = [100]
N_TOP_WORDS = [20]
METRIC = ['cosine']
EMBEDDING_MODELS = ['paraphrase-MiniLM-L6-v2'] #'all-MiniLM-L6-v2'

# SAVE RESULTS 
SAVE_RESULTS_COLS = ["params", "score", "probability"]
SEED_WORDS : Dict[str, List[str]] = {[],[]...,[]} # a list of list
