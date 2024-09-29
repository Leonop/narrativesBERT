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
from sklearn.cluster import MiniBatchKMeans  #GPU-accelerated version KMeans
from model_selection_hpc import vectorize_doc
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from visualize_topic_models import VisualizeTopics as vt
import itertools
from matplotlib import pyplot as plt
import cupy as cp

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
        self.embedding_model = SentenceTransformer(gl.EMBEDDING_MODELS[0], device='cuda')
        # Dimensionality Reduction with UMAP (GPU version from cuML)
        self.umap_model = UMAP(n_components=gl.N_COMPONENTS[0], n_neighbors=gl.N_NEIGHBORS[0], random_state=42, metric=gl.METRIC[0], verbose=True)
        # Clustering with MiniBatchKMeans
        self.cluster_model = MiniBatchKMeans(n_clusters=gl.N_TOPICS[0], random_state=0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        print(f"Using device: {self.device}")

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
                batch_embed, i, i_end = self.process_batch_gpu(i, batch_size, train_docs, self.embedding_model, len(train_docs))
                embeddings[i:i_end, :] = batch_embed

            # Ensure `vocab` is a CountVectorizer object
            vectorizer_model = vectorize_doc(train_docs)
            # Train the BERTopic model on the current fold
            topic_model = self.train_on_fold(train_docs, embeddings, vectorizer_model)
            print(f"Type of topic_model after training: {type(topic_model)}")  # Should output <class 'bertopic.BERTopic'>

            # Evaluate the model on the test set (using the precomputed embeddings for the test set)
            test_embeddings = self.embedding_model.encode(test_docs, show_progress_bar=True, device=self.device)
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
    
    def train_bertopic(self, train_docs, embeddings, vectorizer_model, cluster_model):
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
            hdbscan_model = cluster_model,  # You are using KMeans here, not HDBSCAN, which is fine
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
        
    def model_selection(self, docs_path):
        if os.path.exists(docs_path):
            docs = bt.load_doc(docs_path)
        # Initialize embedding model on GPU
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        
        results = pd.DataFrame(columns=["Number of Topics", "Silhouette Score", "Coherence Score"])
        for num_topics in tqdm(range(10, 200, 10), desc="Training BERTopic models", colour="green", ncols=100):
            cluster_model = KMeans(n_clusters=num_topics, random_state=0)
            document_embeddings = self.embedding_model.encode(docs, show_progress_bar=True, device=self.device)
            # convert the document embeddings to cupy for cuML
            document_embeddings_gpu = cp.array(document_embeddings)
            
            # vectorize documents (assuming this function is compatible with GPU data)
            vectorizer_model = vectorize_doc(docs)
            
            # Initialize BERTopic with the GPU embeddings and cluster model
            topic_model = BERTopic(embedding_model=self.embedding_model, 
                                vectorizer_model=vectorizer_model,
                                hdbscan_model = cluster_model,  # You are using KMeans here, not HDBSCAN, which is fine 
                                umap_model=None,  # Disable UMAP if already handled
                                nr_topics=num_topics,
                                verbose=True)            

            # Train BERTopic
            MS_topic_models = topic_model.fit(docs, embeddings=document_embeddings_gpu)
            
            # Get topic information and save
            topic_info = MS_topic_models.get_topic_info()
            # save the topic information to csv file
            TOPIC_INFO_path = os.path.join(gl.output_folder, f"MS_{num_topics}.csv")
            topic_info.to_csv(TOPIC_INFO_path, index=False)
            
            # save the topic model to a file
            MS_model_path = os.path.join(gl.model_folder, f'MS_bertopic_model_{num_topics}')
            MS_topic_models.save(MS_model_path)
            # compute coherence score
            cscore = self.compute_coherence_score(MS_topic_models, docs)
            
            # get silhouette score (requires CPU)
            topics, _ = MS_topic_models.transform(docs)
            # Transfer embeddings back to CPU
            document_embeddings_cpu = cp.asnumpy(document_embeddings_gpu)
            sscore = silhouette_score(document_embeddings_cpu, topics)
            
            # struture the output in dictionary, with three columns, number of topics, silhouette score, and coherence score
            # write the results to a csv file in append mode
            row = {"Number of Topics": num_topics, "Silhouette Score": sscore, "Coherence Score": cscore}
            row_df = pd.DataFrame([row])
            print(row_df)
            
            # check if the file exists
            if not os.path.exists(gl.MODEL_SELECTION_RESULTS):
                row_df.to_csv(gl.MODEL_SELECTION_RESULTS, index=False)
            else:
                row_df.to_csv(gl.MODEL_SELECTION_RESULTS, mode='a', header=False, index=False)
            results = results.append(row_df, ignore_index=True)
        return results
    
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

        # save it to output fig folder as pdf file
        save_path = os.path.join(gl.output_fig_folder, "model_selection_plot.pdf")  
        plt.savefig(save_path, format="pdf", dpi=600)
        # Show the plot
        plt.tight_layout()
        plt.show()

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
    # bt.save_figures(topic_model)
    # bt.plot_doc_embedding(docs)
    # MODEL SELECTION, IF IT TAKES TOO LONG, PLEASE COMMENT OUT THE FOLLOWING TWO LINES
    results = bt.model_selection(docs_path)
    bt.plot_model_selection(results)
    print("BERTopic model training completed.")
    