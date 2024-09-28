# visualize topics
# Author: Zicheng Xiao
# Date: 2024-09-15
# Purpose: Visualize the topic modeling results, write as a class to call in other scripts or directly run as a script

# Usage: python3 visualize_topic_models.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from bertopic import BERTopic
# from model_selection_hpc import *
import os
import global_options as gl
import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from scipy.cluster import hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
import re
import itertools

class VisualizeTopics:
    def __init__(self):
        self.data_path = os.path.join(gl.data_folder, gl.data_filename)
        self.nrows = gl.NROWS
        self.chunk_size = gl.CHUNK_SIZE
        self.year_filter = gl.YEAR_FILTER
        self.fig_folder = gl.output_fig_folder

    def load_data(self):
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
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The file at path {self.data_path} does not exist.")

        meta = pd.DataFrame()

        try:
            # Use chunksize to limit rows number per iteration
            chunk_reader = pd.read_csv(self.data_path, chunksize=self.chunk_size, nrows=self.nrows)
        except OSError as e:
            print(f"Error reading the file: {e}")
            raise

        # Wrap the chunk reader with tqdm to track progress
        for chunk in tqdm(chunk_reader, total=self.nrows//self.chunk_size):
            chunk["transcriptcreationdate_utc"] = pd.to_datetime(chunk["transcriptcreationdate_utc"])
            chunk["publish_year"] = pd.DatetimeIndex(chunk['transcriptcreationdate_utc']).year
            chunk["publish_month"] = pd.DatetimeIndex(chunk['transcriptcreationdate_utc']).month

            # Select papers published later than the year_filter
            filtered_chunk = chunk[chunk["publish_year"] <= self.year_filter]
            filtered_chunk = filtered_chunk.reset_index()
            meta = pd.concat([meta, filtered_chunk], ignore_index=True)

        # Drop unnecessary columns
        meta = meta.drop(columns=['Unnamed: 0'], errors='ignore')

        # Create a list of documents based on the "componenttextpreview" column
        docs = [str(row['componenttextpreview']) for _, row in meta.iterrows() if row["index"] != self.nrows]

        return meta, docs

    def load_embeddings(self):
        """
        Load the embeddings from a file.

        Parameters:
        embedding_model (str): The name of the Sentence Transformer model.

        Returns:
        np.ndarray: Array of document embeddings.
        """
        # current_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(gl.PROJECT_DIR, "model", f"embeddings_{re.sub('/', '_', str(gl.EMBEDDING_MODELS[0]))}.npy")
        # Check if the file exists before trying to open it
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found at: {file_path}")
        # Load the embeddings
        with open(file_path, "rb") as f:
            embeddings = np.load(f)    
        return embeddings
    
    def visualize_topics(self):
        # Load data from the CSV file
        _, docs = self.load_data()

        # Initialize BERTopic model
        topic_model = BERTopic(language="english")

        # Fit BERTopic to the documents
        topics, probs = topic_model.fit_transform(docs)

        # load embeddings
        embeddings = self.load_embeddings()
        # Use UMAP to reduce the dimensionality of the embeddings to 2D
        umap_model = UMAP(n_neighbors=gl.N_NEIGHBORS[0], n_components=gl.N_COMPONENTS[0], min_dist=gl.MIN_DIST[0], metric=gl.METRIC[0], random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)

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
        # check the fig output folder is exist
        if not os.path.exists(self.fig_folder):
            os.makedirs(self.fig_folder)
        save_path= os.path.join(self.fig_folder, "Topic_pic1.pdf")
        plt.savefig(save_path, format='pdf', dpi=600)
        plt.show()
        return reduced_embeddings
    
    # Updated function
    def plot_and_save_figure(self, df, topic_model, docs):
        """
        Function to plot and save a figure as a PDF for academic publication.
        
        Parameters:
        df (pd.DataFrame): Dataframe containing x, y coordinates, Topic, Length, and color columns.
        topic_model (BERTopic): BERTopic model to get the topic words.
        """
        
        # Adjusted muted color palette suitable for academic journals
        colors = itertools.cycle(['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#e41a1c', '#ffff33', '#a65628', 
                                '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', 
                                '#ffd92f', '#e5c494', '#b3b3b3', '#1f78b4', '#33a02c', '#fb9a99'])

        color_key = {str(topic): next(colors) for topic in set(topic_model.topics_) if topic != -1}

        # Prepare dataframe and filter out outliers or undesired points
        df["Length"] = [len(doc) for doc in docs]
        df = df.loc[df.Topic != "-1"]
        df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
        df["Topic"] = df["Topic"].astype("category")

        # Get centroids of clusters
        mean_df = df.groupby("Topic").mean().reset_index()
        mean_df.Topic = mean_df.Topic.astype(int)
        mean_df = mean_df.sort_values("Topic")

        # Start figure
        fig, ax = plt.subplots(figsize=(16, 16))

        # Convert 'Topic' to string to ensure proper mapping
        df['Topic'] = df['Topic'].astype(str)
        df['color'] = df['Topic'].map(color_key)
        
        # Scatterplot with adjustments for transparency and small marker size
        sns.scatterplot(data=df, x='x', y='y', ax=ax, hue='color', palette=color_key, 
                        alpha=0.1, s=5, legend=False)  # s=5 for small markers, alpha=0.1 for more transparency
        
        # Optionally, add KDE for a smoother, color-blended look similar to the image provided
        sns.kdeplot(data=df, x='x', y='y', ax=ax, fill=True, thresh=0, levels=100, color="k", alpha=0.1)

        # Annotate top topics
        texts, xs, ys = [], [], []
        for _, row in mean_df.iterrows():
            topic = row["Topic"]
            name = " - ".join(list(zip(*topic_model.get_topic(int(topic))))[0][:3])

            if int(topic) <= 50:  # Annotating top 50 topics
                xs.append(row["x"])
                ys.append(row["y"])
                texts.append(plt.text(row["x"], row["y"], name, size=10, ha="center", 
                                    color=color_key[str(int(topic))],
                                    path_effects=[pe.withStroke(linewidth=0.5, foreground="black")]))

        # Adjust annotations to avoid overlapping
        adjust_text(texts, x=xs, y=ys, time_lim=1, 
                    force_text=(0.01, 0.02), force_static=(0.01, 0.02), force_pull=(0.5, 0.5))

        # Check if the output folder exists, create if not
        if not os.path.exists(self.fig_folder):
            os.makedirs(self.fig_folder)
        
        # Define the save path and save as a high-resolution PDF
        save_path = os.path.join(self.fig_folder, gl.TOPIC_SCATTER_PLOT)
        plt.savefig(save_path, format='pdf', dpi=600)
        plt.show()
                
# if __name__ == "__main__":
#     visualizer = VisualizeTopics()
#     visualizer.visualize_topics()
