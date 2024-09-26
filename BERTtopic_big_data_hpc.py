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


# from google.colab import drive
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import global_options as gl
from preprocess_earningscall import NlpPreProcess
import warnings
from sentence_transformers import SentenceTransformer
import collections
from sklearn.feature_extraction.text import CountVectorizer
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

warnings.filterwarnings('ignore')
current_path = os.getcwd()
file_path = os.path.join(current_path, 'data', 'earnings_calls_20231017.csv')
tqdm.pandas()

# BERTopic
# Enableing the GPU for BERTopic
# First, you'll need to enable GPUs for the notebook:
# Navigate to Edit 🡒 Notebook Setting
# Select GPU from the Hardware Accelerator dropdown

class BERTopicGPU(object):
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        # Dimensionality Reduction with UMAP (GPU version from cuML)
        self.umap_model = UMAP(n_components=gl.N_COMPONENTS[0], n_neighbors=gl.N_NEIGHBORS[0], random_state=42, metric=gl.METRIC[0], verbose=True)
        # Clustering with MiniBatchKMeans
        self.cluster_model = MiniBatchKMeans(n_clusters=gl.N_TOPICS[0], random_state=0)
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
    def process_batch_gpu(self, i, batch_size, docs, embedding_model, device, N_):
        i_end = min(i + batch_size, N_)
        batch = docs[i:i_end]
        batch_embed = embedding_model.encode(batch, device=device)
        return batch_embed, i, i_end

    def train_on_fold(self, train_docs, embeddings, vectorizer_model):
        # Ensure embeddings do not have NaN or Inf
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if the shape of embeddings matches the number of documents
        if len(train_docs) != embeddings.shape[0]:
            raise ValueError(f"Number of training docs ({len(train_docs)}) does not match embedding shape ({embeddings.shape[0]}).")
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Number of training documents: {len(train_docs)}")

        # Fit BERTopic with precomputed embeddings and models
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model = self.cluster_model,  # You are using KMeans here, not HDBSCAN, which is fine
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=True
        )

        try:
            # Fit the model and check for any issues
            topic_model.fit(train_docs, embeddings=embeddings)
        except ValueError as e:
            print(f"Error during BERTopic fitting: {e}")
            raise

        return topic_model


    # Main function with cross-validation
    def train_bert_topic_model_cv(self, docs, n_splits=10):
        # Check if a GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Number of documents
        N_ = len(docs)
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        n_cluster = gl.MIN_CLUSTER_SIZE[0]

        # Initialize 10-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Loop over each fold
        fold_scores = []
        for fold, (train_index, test_index) in enumerate(kf.split(docs)):
            print(f"\nProcessing fold {fold + 1}/{n_splits}...")

            # Get training and test documents
            train_docs = [docs[i] for i in train_index]
            test_docs = [docs[i] for i in test_index]

            # Encode training documents (in batches)
            embeddings = np.zeros((len(train_docs), embedding_dim))
            batch_size = gl.BATCH_SIZE
            for i in tqdm(range(0, len(train_docs), batch_size), colour="Blue"):
                batch_embed, i, i_end = self.process_batch_gpu(i, batch_size, train_docs, self.embedding_model, device, len(train_docs))
                embeddings[i:i_end, :] = batch_embed

            # Ensure `vocab` is a CountVectorizer object
            vectorizer_model = vectorize_doc(train_docs)
            # Train the BERTopic model on the current fold
            topic_model = self.train_on_fold(train_docs, embeddings, vectorizer_model)
            print(f"Type of topic_model after training: {type(topic_model)}")  # Should output <class 'bertopic.BERTopic'>

            # Evaluate the model on the test set (using the precomputed embeddings for the test set)
            test_embeddings = self.embedding_model.encode(test_docs, show_progress_bar=True, device=device)
            topics, _ = topic_model.transform(test_docs)
            topic_info = topic_model.get_topic_info()
            # save the topic information to csv file
            TOPIC_INFO_path = os.path.join(gl.output_folder, f"topic_info_{fold}.csv")
            topic_info.to_csv(TOPIC_INFO_path, index=False)
            # Use silhouette score to evaluate the quality of the clusters for this fold
            sscore = silhouette_score(test_embeddings, topics)
            cscore = self.compute_coherence_score(topic_model, train_docs)
            fold_scores.append([sscore, cscore])
            # save the number of topics, and the {fold+1}th fold's silhouette score, and score into a file
            self.save_model_scores(fold + 1, n_cluster, sscore, cscore)
            print(f"Silhouette score for fold {fold + 1}: {sscore}, Coherence score for fold {fold + 1}: {cscore}")
        avg_score = np.mean(fold_scores)
        print(f"\nAverage Silhouette Score across {n_splits} folds: {avg_score}")
            # Compute the average silhouette score and coherence score across all folds
        
        avg_sscore = np.mean([score[0] for score in fold_scores])
        avg_cscore = np.mean([score[1] for score in fold_scores])
        self.save_results(topic_model, n_cluster)
        self.save_model_scores("10 fold Average Score", n_cluster, avg_sscore, avg_cscore)
        return topic_model
    
    def save_results(self, topic_model, n_cluster):
        # Save the BERTopic model
        model_path = os.path.join(gl.model_folder, f'bertopic_model_{n_cluster}')
        # if model_path is not exist, please create it
        if not os.path.exists(gl.model_folder):
            os.makedirs(gl.model_folder)
        topic_model.save(model_path)

    def save_model_scores(self, fold_num, n_cluster, score, cscore):
        # Save the average silhouette score
        score_path = os.path.join(gl.MODEL_SCORES)
        with open(score_path, 'a') as f:
            f.write(f"{fold_num}, {n_cluster}, {score}, {cscore}\n")
        print(f"Average silhouette score saved to {score_path}")
        
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
        
    def save_file(self, data, path, bar_length=100):
        #write the doc to a txt file
        with open(path, 'w') as f:
            with tqdm(total=len(data), desc="Saving data", bar_format="{l_bar}{bar} [time left: {remaining}]", ncols=bar_length, colour="green") as pbar:
                for item in data:
                    f.write("%s\n" % item)
                    pbar.update(1)
                    
    def load_doc(self, path):
        # load the doc from a txt file to a list
        with open(path, 'r') as f:
            return f.readlines()
        
    def plot_topic_scatter(self, topic_model, docs, num_topics):
        # Get the embeddings and reduce them to 2D
        embeddings = topic_model.topic_embeddings_
        reduced_embeddings_2d = self.umap_model.fit_transform(embeddings)
        visual_df = pd.DataFrame({"x": reduced_embeddings_2d[:, 0], "y": reduced_embeddings_2d[:, 1], "Topic": [str(t) for t in topic_model.topics_]})
        # Plot the topics
        vt.plot_and_save_figure(visual_df, topic_model, docs)
            

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
    topic_model = bt.train_bert_topic_model_cv(docs, n_splits = 10)
    bt.save_figures(topic_model)
    # plot the topics
    vt.plot_and_save_figure(topic_model, docs, gl.num_topic_to_plot)
    print("BERTopic model training completed.")
    