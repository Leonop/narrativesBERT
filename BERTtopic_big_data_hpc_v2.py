import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import global_options as gl
from preprocess_earningscall import NlpPreProcess
import warnings
from sentence_transformers import SentenceTransformer
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from bertopic import BERTopic
from sklearn.cluster import MiniBatchKMeans
from model_selection_hpc import vectorize_doc
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from visualize_topic_models import VisualizeTopics as vt
import itertools
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
current_path = os.getcwd()
file_path = os.path.join(current_path, 'data', 'earnings_calls_20231017.csv')
tqdm.pandas()

class BERTopicGPU(object):
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        # Dimensionality Reduction with UMAP (GPU version from cuML)
        self.umap_model = UMAP(n_components=gl.N_COMPONENTS[0], n_neighbors=gl.N_NEIGHBORS[0], random_state=42, metric=gl.METRIC[0], verbose=True)
        # Clustering with MiniBatchKMeans
        self.cluster_model = MiniBatchKMeans(n_clusters=gl.N_TOPICS[0], random_state=0)
        self.hdbscan_model = HDBSCAN(min_samples=gl.MIN_SAMPLES, min_cluster_size=gl.MIN_CLUSTER_SIZE, prediction_data=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize TfidfVectorizer with desired parameters
        self.vectorizer = TfidfVectorizer(
            max_df=0.95,              # Ignore terms with a document frequency higher than this threshold
            min_df=2,                 # Ignore terms with a document frequency lower than this threshold
            stop_words='english',     # Remove English stop words
            ngram_range=(1, 3),       # Consider unigrams and bigrams
            use_idf=True,             # Enable inverse document frequency reweighting
            smooth_idf=True           # Smooth IDF weights by adding one to document frequencies
        )

    def load_data(self):
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at path {file_path} does not exist.")

        # Define the file path and the number of rows to read as a subsample
        nrows = gl.NROWS  # Adjust this number to read a subsample
        chunk_size = gl.CHUNK_SIZE  # Adjust this number to read a subsample
        # Use chunksize to limit rows number per iteration
        meta = pd.DataFrame()
        try:
            chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, nrows=nrows)
        except OSError as e:
            print(f"Error reading the file: {e}")
            raise

        # ANSI escape codes for green color
        GREEN = '\033[92m'
        RESET = '\033[0m'
        # Wrap the chunk reader with tqdm to track progress
        for chunk in tqdm(chunk_reader, total=nrows//chunk_size, bar_format=f'{GREEN}{{l_bar}}{{bar:20}}{{r_bar}}{RESET}'):
            # Select papers published later than 2013
            filtered_chunk = chunk[chunk["year"] <= gl.YEAR_FILTER]
            filtered_chunk = filtered_chunk.reset_index()
            meta = pd.concat([meta, filtered_chunk], ignore_index=True)
        return meta

    def pre_process_text(self, data):
        # Preprocess the text
        nlp = NlpPreProcess()
        data = data[data['speakertypeid'] != 1]
        data['text'] = data[gl.TEXT_COLUMN].astype(str)
        data['text'] = nlp.preprocess_file(data, 'text')
        data['post_date'] = pd.to_datetime(data[gl.DATE_COLUMN])
        data['post_year'] = data['post_date'].dt.year
        data['post_quarter'] = data['post_date'].dt.month
        data['yearq'] = data['post_year'].astype(str) + 'Q' + data['post_quarter'].astype(str)
        data = data.drop(columns = ['Unnamed: 0'])
        docs = [str(row['text']) for _, row in data.iterrows() if len(str(row["text"])) > 30]
        return docs

    def filter_empty_topics(self, topics):
        filtered_topics = {}
        for topic_num, topic_words in topics.items():
            valid_words = [(word, score) for word, score in topic_words if word]  # Remove empty words
            if valid_words:
                filtered_topics[topic_num] = valid_words
        return filtered_topics

    def compute_coherence_score(self, topic_model, texts):
        # Get the top 10 words per topic
        topics = topic_model.get_topics()
        print(f"Number of topics: {len(topics)}")
        filtered_topics = self.filter_empty_topics(topics)
        # Extract topic words into a list of lists
        topics_list = [[word for word, _ in topic_words] for topic_num, topic_words in filtered_topics.items() if topic_num != -1]

        # Ensure texts are tokenized (i.e., a list of lists)
        if isinstance(texts[0], str):
            texts = [doc.split() for doc in texts]  # Simple tokenization if they are in string format

        dictionary = Dictionary(texts)
        
        # Initialize the CoherenceModel
        coherence_model = CoherenceModel(
            topics=topics_list,  # Pass the list of topic words
            texts=texts,
            dictionary=dictionary,  # Create a Gensim dictionary 
            coherence='c_v'
        )
        # Compute the coherence score
        coherence_score = coherence_model.get_coherence()
        return coherence_score

    # Helper function for processing batches
    def process_batch_gpu(self, i, batch_size, docs, embedding_model, N_):
        i_end = min(i + batch_size, N_)
        batch = docs[i:i_end]
        batch_embed = embedding_model.encode(batch, device= self.device)
        return batch_embed, i, i_end
    
    def Bertopic_run(self, docs, embeddings):
        # Ensure embeddings do not have NaN or Inf
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if the shape of embeddings matches the number of documents
        if len(docs) != embeddings.shape[0]:
            raise ValueError(f"Number of training docs ({len(docs)}) does not match embedding shape ({embeddings.shape[0]}).")
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Number of training documents: {len(docs)}")
        
        # use HDBSCAN model to cluster
        batch_size = gl.BATCH_SIZE
        for i in tqdm(range(0, len(docs), batch_size), colour="Blue"):
            batch_embed, i, i_end = self.process_batch_gpu(i, batch_size, docs, self.embedding_model, len(docs))
            embeddings[i:i_end, :] = batch_embed
            
        # use tfidf to vectorize object
        
        # Fit BERTopic with precomputed embeddings and models
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model = self.hdbscan_model,  # You are using KMeans here, not HDBSCAN, which is fine
            vectorizer_model=self.vectorizer,
            calculate_probabilities=True,
            verbose=True
        )
        try:
            # Fit the model and check for any issues
            topic_model.fit(docs, embeddings=embeddings)
        except ValueError as e:
            print(f"Error during BERTopic fitting: {e}")
            raise
        return topic_model
    
    def plot_doc_embedding(self, docs):
        # Get the embeddings and reduce them to 2D
        document_embeddings = self.embedding_model.encode(docs, show_progress_bar=True, device=self.device)
        reduced_embeddings_2d = self.umap_model.fit_transform(document_embeddings)
        vectorizer_model = vectorize_doc(docs)
        topic_model = self.train_on_fold(docs, document_embeddings, vectorizer_model)
        # Ensure that reduced embeddings, topics, and docs have the same length
        if len(reduced_embeddings_2d) != len(docs) or len(topic_model.topics_) != len(docs):
            raise ValueError("Mismatch between the number of embeddings, documents, and topics")

        # Proceed to create the DataFrame if everything matches
        visual_df = pd.DataFrame({
            "x": reduced_embeddings_2d[:, 0],
            "y": reduced_embeddings_2d[:, 1],
            "Topic": topic_model.topics_  # Add topics to the dataframe
        })
        
        # Assigning colors to topics
        colors = itertools.cycle(['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#e41a1c', '#ffff33', '#a65628', 
                                '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', 
                                '#ffd92f', '#e5c494', '#b3b3b3', '#1f78b4', '#33a02c', '#fb9a99'])

        color_key = {str(topic): next(colors) for topic in set(topic_model.topics_) if topic != -1}
        
        # Map colors to the DataFrame based on topic
        visual_df["color"] = visual_df["Topic"].map(lambda x: color_key.get(str(x), '#999999'))  # Default color if topic not found

        # Create a scatter plot
        plt.figure(figsize=(16, 16))
        
        # Scatter plot of the documents with the assigned colors
        plt.scatter(visual_df["x"], visual_df["y"], c=visual_df["color"], alpha=0.4, s=1)  # alpha for transparency, s for small point size

        # Title for the plot
        plt.title("Article-level Nearest Neighbor Embedding", fontsize=16)
        
        # Remove the axis labels for a cleaner look
        plt.axis("off")
        # Optionally, save the plot as a high-resolution PNG file
        save_path = os.path.join(gl.output_fig_folder, "article_level_embedding_plot.pdf")
        plt.savefig(save_path, format="pdf", dpi=600)
        # Show the plot
        plt.show()
        

    def plot_model_selection(self, df):
        """
        Function to plot model selection data with two curves: 
        Silhouette Score and Coherence Score.
        
        Parameters:
        df (pd.DataFrame): Dataframe containing 'Number of Topics', 'Silhouette Score', and 'Coherence Score'
        """
        plt.figure(figsize=(8, 6))

        # Plot Cross-Validation (Silhouette Score) in blue
        plt.plot(df['Number of Topics'], df['Silhouette Score'], label='Cross-Validation', color='blue')

        # Plot Bayes-Factor (Coherence Score) in red
        plt.plot(df['Number of Topics'], df['Coherence Score'], label='Bayes-Factor', color='red')

        # Highlight the maximum point of Cross-Validation (Silhouette Score)
        max_silhouette_idx = df['Silhouette Score'].idxmax()
        plt.scatter(df['Number of Topics'].iloc[max_silhouette_idx], df['Silhouette Score'].iloc[max_silhouette_idx], 
                    color='red', s=100, label='Max Silhouette', marker='^')

        # Adding labels and title
        plt.title("Model Selection", fontsize=16, fontweight='bold')
        plt.xlabel("Number of Topics (K)", fontsize=12)
        plt.ylabel("Score", fontsize=12)

        # Adding grid and legend
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
        plt.legend(loc='best')

        # Show the plot
        plt.tight_layout()
        plt.show()
        # save it to output fig folder as pdf file
        save_path = os.path.join(gl.output_fig_folder, "model_selection_plot.pdf")  
        plt.savefig(save_path, format="pdf", dpi=600)

    def save_file(self, data, path, bar_length=100):
        #write the doc to a txt file
        with open(path, 'w') as f:
            with tqdm(total=len(data), desc="Saving data", bar_format="{l_bar}{bar} [time left: {remaining}]", ncols=bar_length, colour="green") as pbar:
                for item in data:
                    f.write("%s\n" % item)
                    pbar.update(1)
                    
    def save_figures(self, topic_model):
        # Save the visualization
        visualization_path = os.path.join(gl.output_fig_folder, f'bertopic{gl.num_topic_to_plot}.pdf')
        fig = topic_model.visualize_barchart(top_n_topics=gl.num_topic_to_plot)
        fig.write_image(visualization_path)
        fig1 = topic_model.visualize_topics()
        fig1.write_image(visualization_path.replace('.pdf', '_intertopic_distance_map.pdf'))
        fig2 = topic_model.visualize_heatmap()
        fig2.write_image(visualization_path.replace('.pdf', '_heatmap.pdf'))
        fig3 = topic_model.visualize_hierarchy()
        fig3.write_image(visualization_path.replace('.pdf', '_hierarchy.pdf'))
        print(f"Visualization saved to {visualization_path}")

if __name__ == "__main__":
    bt = BERTopicGPU()
    meta = bt.load_data()
    docs_path = os.path.join(gl.output_folder, 'preprocessed_docs.txt')
    # if the processed doc file is exist, please load it
    if os.path.exists(docs_path):
        docs = bt.load_doc(docs_path)
    else:
        docs = bt.pre_process_text(meta)
        # save docs to a file
        bt.save_file(docs, docs_path, bar_length=100)
    # topic_model = bt.train_bert_topic_model_cv(docs, n_splits = 10)
    topic_model = bt.Bertopic_run(docs)
    bt.save(topic_model)
    bt.plot_doc_embedding(docs)
    print("BERTopic model training completed.")