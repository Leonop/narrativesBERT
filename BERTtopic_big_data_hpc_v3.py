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
# from model_selection_hpc import vectorize_doc
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from visualize_topic_models import VisualizeTopics as vt
import itertools
from matplotlib import pyplot as plt
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech

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
        # self.cluster_model = MiniBatchKMeans(n_clusters=gl.N_TOPICS[0], random_state=0)
        self.hdbscan_model = HDBSCAN(min_samples=gl.MIN_SAMPLES[0], min_cluster_size=gl.MIN_CLUSTER_SIZE[0], prediction_data=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize TfidfVectorizer with desired parameters
        self.vectorizer = TfidfVectorizer(
            max_df=gl.MAX_DF[0],              # Ignore terms with a document frequency higher than this threshold
            min_df=gl.MIN_DF[0],                 # Ignore terms with a document frequency lower than this threshold
            stop_words='english',     # Remove English stop words
            ngram_range=(1, 1),       # Consider unigrams and bigrams
            use_idf=True,             # Enable inverse document frequency reweighting
            smooth_idf=True           # Smooth IDF weights by adding one to document frequencies
        )
        
        self.representation_model = {
                "KeyBERT": KeyBERTInspired(),
                "MMR": MaximalMarginalRelevance(diversity=0.3),
                "POS": PartOfSpeech("en_core_web_sm")
            }

    def load_data(self):
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at path {file_path} does not exist.")

        # Define the file path and the number of rows to read as a subsample
        chunk_size = gl.CHUNK_SIZE  # Adjust this number to read a subsample
        # Use chunksize to limit rows number per iteration
        meta = pd.DataFrame()
        try:
            chunk_reader = pd.read_csv(
                                        file_path, 
                                       chunksize=chunk_size, 
                                       skiprows=range(1, gl.START_ROWS+1),
                                       nrows=gl.NROWS  # Adjust this number to read a subsample
                                       )
        except OSError as e:
            print(f"Error reading the file: {e}")
            raise

        # ANSI escape codes for green color
        GREEN = '\033[92m'
        RESET = '\033[0m'
        # Wrap the chunk reader with tqdm to track progress
        for chunk in tqdm(chunk_reader, total=gl.NROWS//chunk_size, bar_format=f'{GREEN}{{l_bar}}{{bar:20}}{{r_bar}}{RESET}'):
            filtered_chunk = chunk[(chunk["year"] <= gl.YEAR_FILTER) & (chunk["year"] >= gl.START_YEAR)] # Filter by START_YEAR and YEAR_FILTER
            filtered_chunk = filtered_chunk.reset_index()
            filtered_chunk = filtered_chunk.sort_values(by='isdelayed_flag', ascending=False).drop_duplicates(subset=gl.UNIQUE_KEYS, keep='first')
            meta = pd.concat([meta, filtered_chunk], ignore_index=True)       
        return meta

    def pre_process_text(self, data):
        # Preprocess the text
        nlp = NlpPreProcess()
        data = data[data['speakertypeid'] != 1]
        data['text'] = data[gl.TEXT_COLUMN].astype(str)
        data['post_date'] = pd.to_datetime(data[gl.DATE_COLUMN])
        data['post_year'] = data['post_date'].dt.year
        data['post_quarter'] = data['post_date'].dt.month
        data['yearq'] = data['post_year'].astype(str) + 'Q' + data['post_quarter'].astype(str)
        data = data.drop(columns = ['Unnamed: 0'])
        data['text'] = nlp.preprocess_file(data, 'text')
        data = data.drop_duplicates(subset='text', keep='first')
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
    
    def Bertopic_run(self, docs):
        print("numpy imported:", 'np' in globals())
        print(type(docs))
        print(len(docs))
        # Initialize an empty array for embeddings
        embeddings = np.zeros((len(docs), self.embedding_model.get_sentence_embedding_dimension()))
        
        # Process documents in batches to compute embeddings
        batch_size = gl.BATCH_SIZE
        for i in tqdm(range(0, len(docs), batch_size), colour="Blue"):
            batch_embed, i, i_end = self.process_batch_gpu(i, batch_size, docs, self.embedding_model, len(docs))
            embeddings[i:i_end, :] = batch_embed
        
        # Ensure embeddings do not have NaN or Inf
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if the shape of embeddings matches the number of documents
        if len(docs) != embeddings.shape[0]:
            raise ValueError(f"Number of training docs ({len(docs)}) does not match embedding shape ({embeddings.shape[0]}).")
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Number of training documents: {len(docs)}")
        # Add these debug prints in both environments
        print("Local/Cloud Environment Check:")
        print("Embeddings type:", type(embeddings))
        print("Embeddings shape:", embeddings.shape)
        print("Embeddings dtype:", embeddings.dtype)
        print("Number of documents:", len(docs))
        print("SEED_TOPICS length:", len(gl.SEED_TOPICS))
        print("Sample SEED_TOPICS shape:", [len(topic) for topic in gl.SEED_TOPICS[:3]])

        # Check for NaN or infinite values
        print("Has NaN:", np.isnan(embeddings).any())
        print("Has Inf:", np.isinf(embeddings).any())    
        # Fit BERTopic with precomputed embeddings and models
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model = self.hdbscan_model,  
            vectorizer_model = self.vectorizer,
            calculate_probabilities=True,
            top_n_words=gl.TOP_N_WORDS[0],
            verbose=True,
            nr_topics=gl.NR_TOPICS[0],
            seed_topic_list=gl.SEED_TOPICS,
            representation_model=self.representation_model
        )
        try:
            # Fit the model and check for any issues
            topic_model.fit(docs, embeddings=embeddings)
        except ValueError as e:
            print(f"Error during BERTopic fitting: {e}")
            raise
        topic_model.save(os.path.join(gl.model_folder, f"bertopic_model_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}"))
        return topic_model
    

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
        fig1.write_image(visualization_path.replace('.pdf', f'_intertopic_distance_map_{gl.NROWS}.pdf'))
        fig2 = topic_model.visualize_heatmap()
        fig2.write_image(visualization_path.replace('.pdf', f'_heatmap_{gl.NROWS}.pdf'))
        fig3 = topic_model.visualize_hierarchy()
        fig3.write_image(visualization_path.replace('.pdf', f'_hierarchy_{gl.NROWS}.pdf'))
        print(f"Visualization saved to {visualization_path}")

    def load_doc(self, path):
        # load the doc from a txt file to a list
        with open(path, 'r') as f:
            return f.readlines()
        
    def save_topic_keywords(self, topic_model):
        # Get topic information and save
        topic_info = topic_model.get_topic_info()
        num_topic = len(topic_info)
        # save the topic information to csv file
        TOPIC_INFO_path = os.path.join(gl.output_folder, f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}.csv")
        topic_info.to_csv(TOPIC_INFO_path, index=False)
        


    
if __name__ == "__main__":
    bt = BERTopicGPU()
    docs_path = os.path.join(gl.output_folder, 'preprocessed_docs.txt')
    # if the processed doc file is exist, please load it
    print(docs_path)
    if os.path.exists(docs_path):
        print("Reading preprocessed docs from preprocessed_docs.txt")
        docs = bt.load_doc(docs_path)
        docs = list(set(docs))
    else:
        meta = bt.load_data()
        docs = bt.pre_process_text(meta)
        # save docs to a file
        bt.save_file(docs, docs_path, bar_length=100)
    # topic_model = bt.train_bert_topic_model_cv(docs, n_splits = 10)
    topic_model = bt.Bertopic_run(docs)
    bt.save_topic_keywords(topic_model)
    # bt.plot_doc_embedding(docs)
    bt.save_figures(topic_model)
    print("BERTopic model training completed.")