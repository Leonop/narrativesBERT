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
from bertopic.representation import PartOfSpeech
# from bertopic.vectorizers import ClassTfidfTransformer # perofrmance improvement with seed words
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
# Optional: For monitoring memory usage
import psutil
import re
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import global_options
from itertools import product
from visualize_topic_models import VisualizeTopics
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.decomposition import IncrementalPCA
# from bertopic.vectorizers import OnlineCountVectorizer

# Global Variables
current_path = global_options.PROJECT_DIR
data_path = os.path.join(global_options.data_folder, global_options.data_filename)
NROWS = global_options.NROWS # number of rows to read from the csv file
CHUNK_SIZE = global_options.CHUNK_SIZE # number of rows to read at a time
YEAR_FILTER = global_options.YEAR_FILTER # filter the data based on the year

# how to import seedwords from global_variables.py
# from global_variables import seedwords


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
    vocab = [word for word, frequency in vocab.items() if frequency > 15]; 
    vocab = set([word for words in global_options.SEED_WORDS for word in words] + list(vocab))
    # remove word are digits only and words with length less than 3
    vocab = [word for word in vocab if not word.isdigit() and len(word) > 2]
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
    
    topics, _prob = topic_model.fit_transform(docs, embeddings)
    
    # Evaluate coherence (using silhouette score as a placeholder for coherence)
    if len(set(topics)) > 1:  # Silhouette score requires more than one cluster
        score = silhouette_score(embeddings, topics)
    else:
        score = -1  # Invalid configuration with no clusters
    
    # Optional: Check memory usage
    memory_usage = psutil.virtual_memory().percent
    print(f"Memory usage: {memory_usage}%")
    
    return score, topics, _prob, topic_model

def save_csv(results, filename):
    """
    Save the results to a CSV file.

    Parameters:
    results (list): List of tuples containing parameters and scores.
    file_path (str): The file path to save the results.
    """
    file_path = os.path.join(current_path, "output", filename)
    if not os.path.exists(os.path.join(current_path, "output")):
        os.makedirs(os.path.join(current_path, "output"))
    df = pd.DataFrame(results, columns=global_options.SAVE_RESULTS_COLS)
    # save the results to a csv file in append mode
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)
    
# Main function
def main():
    # Embedding models to iterate over
    embedding_models = global_options.EMBEDDING_MODELS
    
    for embedding_model in embedding_models:
        # Load data
        print(f"Loading data for embedding model: {embedding_model}")
        meta, docs = load_data(data_path, nrows=global_options.NROWS, 
                               chunk_size=global_options.CHUNK_SIZE, year_filter=global_options.YEAR_FILTER)

        # Generate embeddings
        print(f"Generating embeddings for {embedding_model}")
        embeddings = generate_embeddings(docs, embedding_model)
        
        # Save embeddings
        print(f"Saving embeddings for {embedding_model}")
        save_embeddings(embeddings, embedding_model)

    # Define parameter grid for optimization
    param_grid = {
        'n_neighbors': global_options.N_NEIGHBORS,
        'n_components': global_options.N_COMPONENTS,
        'min_dist': global_options.MIN_DIST,
        'metric': global_options.METRIC,
        'min_samples': global_options.MIN_SAMPLES,
        'min_cluster_size': global_options.MIN_CLUSTER_SIZE,
        'embedding_model': global_options.EMBEDDING_MODELS  # List of embedding models
    }
    # Vectorize the documents
    print("Vectorizing documents...")
    vectorizer_model = vectorize_doc(docs)

    # Create a list of all parameter combinations
    print("Generating parameter combinations...")
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Test each configuration and store results
    best_params = None
    best_score = -1
    results = []

    print("Starting model evaluation with different parameter combinations...")
    
    # Adding a progress bar for the parameter search
    for params in tqdm(param_combinations, desc="Evaluating params"):
        print(f"Testing params: {params}")
        
        # Evaluate the topic model with the current parameters
        try:
            score, topics, probability, topic_model = evaluate_topic_model(params, docs, vectorizer_model)
        except Exception as e:
            print(f"Error evaluating params: {params}, Error: {e}")
            continue

        # Update best parameters if a higher score is found
        if score > best_score:
            best_score = score
            best_params = params

        print(f"Score: {score}, Best score: {best_score}, Best params: {best_params}, Probability: {probability}")
        results.append((params, score, probability))

    # Save results to a CSV file
    print("Saving model selection results...")
    save_csv(results, "model_selection_results.csv")
    # visualize topic_models.py
    vt = VisualizeTopics()
    reduced_embeddings_2d = vt.visualize_topics()
    _df = pd.DataFrame({"x": reduced_embeddings_2d[:, 0], "y": reduced_embeddings_2d[:, 1], "Topic": [str(t) for t in topic_model.topics_]})
    vt.plot_and_save_figure(_df, topic_model, docs)
    vt.hirachical_cluster_visualization(docs, topic_model)
if __name__ == "__main__":
    main()