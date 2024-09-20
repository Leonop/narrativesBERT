import os
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic import BERTopic
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='bertopic_online.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the function to determine optimal number of topics
def determine_optimal_num_topics(embeddings, 
                                 min_clusters=5, 
                                 max_clusters=100, 
                                 step=5, 
                                 sample_size=10000, 
                                 random_state=42, 
                                 plot=True):
    inertias = []
    silhouette_scores = []
    cluster_range = range(min_clusters, max_clusters + 1, step)
    
    print("Evaluating cluster numbers...")
    
    for k in tqdm(cluster_range, desc="Determining optimal clusters"):
        kmeans = MiniBatchKMeans(n_clusters=k, 
                                 random_state=random_state, 
                                 batch_size=1000)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        
        # To speed up, sample a subset for silhouette_score
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = kmeans.predict(sample_embeddings)
            score = silhouette_score(sample_embeddings, sample_labels)
        else:
            labels = kmeans.labels_
            score = silhouette_score(embeddings, labels)
        
        silhouette_scores.append(score)
    
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia', color=color)
        ax1.plot(cluster_range, inertias, marker='o', color=color, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel('Silhouette Score', color=color)
        ax2.plot(cluster_range, silhouette_scores, marker='x', color=color, label='Silhouette Score')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title('Clustering Evaluation Metrics')
        plt.show()
    
    # Determine optimal k based on highest Silhouette Score
    optimal_k_silhouette = cluster_range[np.argmax(silhouette_scores)]
    
    # Optionally, determine based on Elbow Method
    inertias_diff = np.diff(inertias)
    inertias_diff2 = np.diff(inertias_diff)
    elbow_point = np.argmax(inertias_diff2) + min_clusters
    
    print(f"Optimal number of clusters by Silhouette Score: {optimal_k_silhouette}")
    print(f"Optimal number of clusters by Elbow Method: {elbow_point}")
    
    # Return the optimal k based on Silhouette Score
    return optimal_k_silhouette

# Load and preprocess your data
file_path = 'your_file.csv'  # Replace with your actual file path

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

nrows = 1000000
chunk_size = 10000

meta = pd.DataFrame()

try:
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, nrows=nrows)
except OSError as e:
    print(f"Error reading the file: {e}")
    raise

for chunk in tqdm(chunk_reader, total=nrows // chunk_size, desc="Loading Data"):
    chunk["transcriptcreationdate_utc"] = pd.to_datetime(chunk["transcriptcreationdate_utc"], errors='coerce')
    chunk["publish_year"] = chunk["transcriptcreationdate_utc"].dt.year
    chunk["publish_month"] = chunk["transcriptcreationdate_utc"].dt.month
    chunk["publish_quarter"] = chunk["transcriptcreationdate_utc"].dt.quarter
    chunk["publish_yearq"] = chunk["publish_year"].astype(str) + "Q" + chunk["publish_quarter"].astype(str)
    filtered_chunk = chunk[(chunk["publish_year"] >= 2010) & (chunk["publish_year"] <= 2022)]
    filtered_chunk = filtered_chunk.reset_index(drop=True)
    meta = pd.concat([meta, filtered_chunk], ignore_index=True)

meta = meta.drop(columns=['Unnamed: 0'], errors='ignore')

# Predefine vocabulary using a large sample or the entire filtered data
sample_size_vocabulary = min(100000, len(meta))  # Adjust based on memory
sample_docs = meta['componenttextpreview'].astype(str).iloc[:sample_size_vocabulary].tolist()

# Initialize a standard CountVectorizer to build the vocabulary
standard_vectorizer = CountVectorizer(
    stop_words="english",
    max_features=10000  # Adjust based on your dataset
)
standard_vectorizer.fit(sample_docs)

# Extract the vocabulary
predefined_vocabulary = standard_vectorizer.vocabulary_

# Initialize online sub-models with predefined vocabulary
vectorizer_model = OnlineCountVectorizer(
    stop_words="english",
    decay=0.01,
    vocabulary=predefined_vocabulary
)

umap_model = IncrementalPCA(n_components=5)
cluster_model = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=1000)

# Initialize BERTopic with online sub-models
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model
)

# Initialize a list to store all topics
topics_ = []

# Determine the number of unique publish_yearq
number_of_groups = meta["publish_yearq"].nunique()

# Iterate over each year-quarter group with progress tracking
for yearq, yearq_chunk in tqdm(
    meta.groupby("publish_yearq"),
    total=number_of_groups,
    desc="Fitting BERTopic",
    colour='blue'
):
    # Extract documents, ensuring 'index' exists
    if "index" in yearq_chunk.columns:
        chunk_docs = yearq_chunk.loc[yearq_chunk["index"] != 1000000, 'componenttextpreview'].astype(str).tolist()
    else:
        # If 'index' doesn't exist, include all documents
        chunk_docs = yearq_chunk['componenttextpreview'].astype(str).tolist()
    
    # Incrementally fit the BERTopic model
    topic_model.partial_fit(chunk_docs)
    
    # Assign topics to the current chunk
    topics, probs = topic_model.transform(chunk_docs)
    
    # Extend the topics_ list
    topics_.extend(topics)
    
    # Log progress
    logging.info(f"Fitted BERTopic on year-quarter: {yearq} with {len(chunk_docs)} documents.")

# Assign the collected topics to the model
topic_model.topics_ = topics_

# Extract embeddings
embeddings = topic_model.embeddings_

# Determine the optimal number of topics
optimal_num_topics = determine_optimal_num_topics(
    embeddings=embeddings,
    min_clusters=5,
    max_clusters=100,
    step=5,
    sample_size=10000,
    random_state=42,
    plot=True
)

print(f"Recommended number of topics: {optimal_num_topics}")

# Re-initialize MiniBatchKMeans with the optimal number of clusters
optimal_cluster_model = MiniBatchKMeans(
    n_clusters=optimal_num_topics, 
    random_state=42, 
    batch_size=1000
)

# Re-initialize BERTopic with the updated clustering model
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=optimal_cluster_model,
    vectorizer_model=vectorizer_model
)

# Optionally, re-fit the model on all documents if feasible
# Otherwise, continue with incremental fitting using the new cluster model
# Here, for demonstration, we'll assume re-fitting is not feasible due to data size

# Save the model
topic_model.save("online_bertopic_model")

# To load the model later
# loaded_model = BERTopic.load("online_bertopic_model")

# Access and print topic information
print("Total Topics:", len(topic_model.get_topic_info()))
print("Sample Topic Info:")
print(topic_model.get_topic_info().head())

# Optional: Visualize topics
# fig = topic_model.visualize_topics()
# fig.show()
