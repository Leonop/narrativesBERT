# visualize topics
# Author: Zicheng Xiao
# Date: 2024-09-15
# Purpose: Visualize the topic modeling results, write as a class to call in other scripts or directly run as a script

# Usage: python3 visualize_topic_models.py

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from model_selection_hpc import *
import os
import global_options

class VisualizeTopics:
    def __init__(self, data_path, nrows, chunk_size, year_filter):
        self.data_path = data_path
        self.nrows = nrows
        self.chunk_size = chunk_size
        self.year_filter = year_filter

    def visualize_topics(self):
        # Load data from the CSV file
        data, docs = load_data(self.data_path, self.nrows, self.chunk_size, self.year_filter)

        # Initialize BERTopic model
        topic_model = BERTopic(language="english")

        # Fit BERTopic to the documents
        topics, probs = topic_model.fit_transform(docs)

        # Use UMAP to reduce the dimensionality of the embeddings to 2D
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(topic_model.embedding_model.encode(docs))

        # Prepare a color map for different topics
        unique_topics = np.unique(topics)
        topic_colors = plt.cm.get_cmap('tab20', len(unique_topics))

        # Plot the reduced embeddings, colored by topic
        plt.figure(figsize=(10, 8))
        for topic in unique_topics:
            indices = np.where(topics == topic)
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                        color=topic_colors(topic), label=f"Topic {topic}", alpha=0.7, s=50)

        plt.title("UMAP visualization of topic clusters")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
        plt.show()
        
if __name__ == "__main__":
    data_path = os.path.join(global_options.data_folder, global_options.data_filename)
    nrows = 10000
    chunk_size = 1000
    year_filter = 2020

    visualizer = VisualizeTopics(data_path, nrows, chunk_size, year_filter)
    visualizer.visualize_topics()
