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
import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from scipy.cluster import hierarchy as sch

class VisualizeTopics:
    def __init__(self, nrows, chunk_size, year_filter):
        self.data_path = os.path.join(global_options.data_folder, global_options.data_filename)
        self.nrows = global_options.NROWS
        self.chunk_size = global_options.CHUNK_SIZE
        self.year_filter = global_options.YEAR_FILTER
        self.fig_folder = global_options.output_fig_folder

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
        
    def plot_and_save_figure(self, df, mean_df, topic_model, color_key):
        """
        Function to plot and save a figure as a PDF.
        
        Parameters:
        df (pd.DataFrame): Dataframe containing x, y coordinates, Topic, Length, and color columns.
        mean_df (pd.DataFrame): Dataframe containing Topic and x, y coordinates for annotation.
        topic_model (BERTopic): BERTopic model to get the topic words.
        color_key (dict): Dictionary mapping topics to colors.
        save_path (str): Path to save the figure as a PDF.
        """
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # Convert 'Topic' to string to ensure proper mapping
        df['Topic'] = df['Topic'].astype(str)
        df['color'] = df['Topic'].map(color_key)
        
        # Scatterplot
        sns.scatterplot(data=df, x='x', y='y', ax=ax, hue='color', alpha=0.4, s=10, sizes=(0.4, 10), size="Length", legend=False)
        
        # Annotate top 50 topics
        texts, xs, ys = [], [], []
        for _, row in mean_df.iterrows():
            topic = row["Topic"]
            name = " - ".join(list(zip(*topic_model.get_topic(int(topic))))[0][:3])

            if int(topic) <= 50:
                xs.append(row["x"])
                ys.append(row["y"])
                texts.append(plt.text(row["x"], row["y"], name, size=10, ha="center", color=color_key[str(int(topic))],
                                    path_effects=[pe.withStroke(linewidth=0.5, foreground="black")]))
        
        # Adjust annotations such that they do not overlap
        adjust_text(texts, x=xs, y=ys, time_lim=1, force_text=(0.01, 0.02), force_static=(0.01, 0.02), force_pull=(0.5, 0.5))
        
        # check the fig output folder is exist
        if not os.path.exists():
            os.makedirs(self.fig_folder)
        save_path= os.path.join(self.fig_folder, "visualization_topics.pdf")
        # Save the plot as a PDF
        plt.savefig(save_path, format='pdf', dpi=600)
        plt.show()
        
    def hirachical_cluster_visualization(self, data, docs, topic_model):
        """
        Function to visualize the hierarchical clustering of the topics.
        
        Parameters:
        data (pd.DataFrame): Dataframe containing the data.
        docs (List[str]): List of documents.
        topic_model (BERTopic): BERTopic model.
        """
        # The resulting hierarchical_topics is a dataframe in which merged topics are described.
        #Hierachical topics
        linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
        hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
        topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        # Visualize the hierarchical topics
        fig = topic_model.visualize_hierarchy()
        # Plot and save the figure
        # check if the fig folder is exist in output folder
        if not os.path.exists(global_options.output_fig_folder):
            os.makedirs(global_options.output_fig_folder)
        fig.savefig(os.path.join(global_options.output_fig_folder, "visualization_hierarchical_topics.pdf"), format='pdf', dpi=600)
                
if __name__ == "__main__":
    data_path = os.path.join(global_options.data_folder, global_options.data_filename)
    nrows = global_options.NROWS
    chunk_size = global_options.CHUNK_SIZE
    year_filter = global_options.YEAR_FILTER

    visualizer = VisualizeTopics(data_path, nrows, chunk_size, year_filter)
    visualizer.visualize_topics()
