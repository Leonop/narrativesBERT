# Model Selection
# Author: Zicheng Xiao
# Date: 2014-09-15
# Purpose: This script is used to select the best parameters for bertopic model
# Usage: python3 model_selection_hpc.py
# Python version: 3.8.19
# Virtualenv: Cuda_env
# Import necessary libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
# Optional: For monitoring memory usage
import psutil
import re
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Global Variables
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'earnings_calls_20231017.csv')
NROWS = 10000 # number of rows to read from the csv file
CHUNK_SIZE = 1000000 # number of rows to read at a time
YEAR_FILTER = 2025 # filter the data based on the year

import os
import pandas as pd
from tqdm import tqdm

def load_data(file_path, nrows=NROWS, chunk_size=CHUNK_SIZE, year_filter=YEAR_FILTER):
    """
    Load data from a CSV file in chunks and filter based on the year.

    Parameters:
    file_path (str): The file path to the CSV file.
    nrows (int): The total number of rows to read as a subsample.
    chunk_size (int): The number of rows to read per chunk.
    year_filter (int): The year threshold for filtering.

    Returns:
    pd.DataFrame: Filtered and concatenated DataFrame.
    list: List of document texts.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path {file_path} does not exist.")

    meta = pd.DataFrame()

    try:
        # Use chunksize to limit rows number per iteration
        chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, nrows=nrows)
    except OSError as e:
        print(f"Error reading the file: {e}")
        raise

    # Wrap the chunk reader with tqdm to track progress
    for chunk in tqdm(chunk_reader, total=nrows//chunk_size):
        chunk["transcriptcreationdate_utc"] = pd.to_datetime(chunk["transcriptcreationdate_utc"])
        chunk["publish_year"] = pd.DatetimeIndex(chunk['transcriptcreationdate_utc']).year
        chunk["publish_month"] = pd.DatetimeIndex(chunk['transcriptcreationdate_utc']).month

        # Select papers published later than the year_filter
        filtered_chunk = chunk[chunk["publish_year"] <= year_filter]
        filtered_chunk = filtered_chunk.reset_index()
        meta = pd.concat([meta, filtered_chunk], ignore_index=True)

    # Drop unnecessary columns
    meta = meta.drop(columns=['Unnamed: 0'], errors='ignore')

    # Create a list of documents based on the "componenttextpreview" column
    docs = [str(row['componenttextpreview']) for _, row in meta.iterrows() if row["index"] != nrows]

    return meta, docs

def generate_embeddings(docs, embedding_model):
    """
    Generate embeddings for a list of documents using a Sentence Transformer model.

    Parameters:
    docs (list): List of document texts.
    embedding_model (str): The name of the Sentence Transformer model.
    For example: 'sentence-transformers/paraphrase-MiniLM-L6-v2' or 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
    np.ndarray: Array of document embeddings.
    """
    # Load the Sentence Transformer model
    model = SentenceTransformer(embedding_model)

    # Generate embeddings for the documents
    embeddings = model.encode(docs, show_progress_bar=True)

    return embeddings

def save_embeddings(embeddings, embedding_model):
    """
    Save the embeddings to a file.

    Parameters:
    embeddings (np.ndarray): Array of document embeddings.
    save_path (str): The file path to save the embeddings.
    """
    with open(os.path.join(current_path, "model", f"embeddings_{re.sub('/', '_', str(embedding_model))}.npy"), "wb") as f:
        np.save(f, embeddings)

def load_embeddings(embedding_model):
    """
    Load the embeddings from a file.

    Parameters:
    embedding_model (str): The name of the Sentence Transformer model.

    Returns:
    np.ndarray: Array of document embeddings.
    """
    # current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path, "model", f"embeddings_{re.sub('/', '_', str(embedding_model))}.npy")
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found at: {file_path}")
    # Load the embeddings
    with open(file_path, "rb") as f:
        embeddings = np.load(f)
    
    return embeddings

def vectorize_doc(docs):
    """
    Preprocess the vocabulary before training the model.

    Parameters:
    docs (list): List of document texts.

    Returns:
    list: List of preprocessed document texts.
    """
    # Extract vocab to be used in BERTopic
    vocab = collections.Counter()
    tokenizer = CountVectorizer().build_tokenizer()
    for doc in tqdm(docs, total=len(docs)):
        vocab.update(tokenizer(doc))
    vocab = [word for word, frequency in vocab.items() if frequency > 15]; len(vocab)

    vectorize_model = CountVectorizer(ngram_range=(1, 2), vocabulary=vocab, stop_words="english")
    return vectorize_model
    
# Function to train and evaluate BERTopic model
def evaluate_topic_model(params, docs, vectorizer_model):

    # UMAP model to reduce the dimensionality of the embeddings
    umap_model = UMAP(
        n_neighbors=params['n_neighbors'], 
        n_components=params['n_components'], 
        min_dist=params['min_dist'], 
        metric=params['metric'], 
        random_state=42
    )

    # HDBSCAN model to cluster the similar documents
    hdbscan_model = HDBSCAN(
        min_samples=params['min_samples'], 
        min_cluster_size=params['min_cluster_size'], 
        prediction_data=True
    )

    embeddings = load_embeddings(params['embedding_model'])

    # BERTopic model
    topic_model = BERTopic(
        embedding_model=params['embedding_model'], 
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model, 
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    topics, _ = topic_model.fit_transform(docs, embeddings)
    
    # Evaluate coherence (using silhouette score as a placeholder for coherence)
    if len(set(topics)) > 1:  # Silhouette score requires more than one cluster
        score = silhouette_score(embeddings, topics)
    else:
        score = -1  # Invalid configuration with no clusters
    
    # Optional: Check memory usage
    memory_usage = psutil.virtual_memory().percent
    print(f"Memory usage: {memory_usage}%")
    
    return score

if __name__ == "__main__":
    embedding_models = ['paraphrase-MiniLM-L6-v2', 'all-MiniLM-L6-v2']
    for embedding_model in embedding_models:
        # Load data
        meta, docs = load_data(data_path, nrows=NROWS, chunk_size=CHUNK_SIZE, year_filter=YEAR_FILTER)
        # Generate embeddings
        embeddings = generate_embeddings(docs, embedding_model)
        # Save embeddings
        save_embeddings(embeddings, embedding_model)

    # Define parameter grid for optimization
    param_grid = {
        'n_neighbors': [10, 15, 30],
        'n_components': [5, 10, 15],
        'min_dist': [0.0, 0.1, 0.5],
        'metric': ['cosine', 'euclidean'],
        'min_samples': [5, 10, 20],
        'min_cluster_size': [20, 50, 100],
        'embedding_model': ['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'],  # Smaller models
    }
    vectorizer_model = vectorize_doc(docs)
    # Create a list of all parameter combinations
    from itertools import product
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    # Test each configuration and store results
    best_params = None
    best_score = -1
    results = []
    for params in param_combinations:
        print(f"Testing params: {params}")
        score = evaluate_topic_model(params, docs, vectorizer_model)
        if score > best_score:
            best_score = score
            best_params = params
        print(f"Score: {score}, Best score: {best_score}, Best params: {best_params}")
        results.append((params, score))