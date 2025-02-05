#!/usr/bin/env python
"""
BERTtopic_big_data_hpc_v7.py

This refactored script improves performance and design by splitting the overall
pipeline into modular components. It includes:
  - DataHandler: Loading and preprocessing data.
  - EmbeddingGenerator: Generating document embeddings with SentenceTransformer.
  - TopicModeler: Training BERTopic, saving topic keywords and visualization figures.

Configuration and constants are assumed to be provided via the imported module 'global_options'.
"""

import os
import sys
import time
import traceback
import logging
from multiprocessing import cpu_count, Pool

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from sklearn.decomposition import PCA
import joblib
import re
from openai import OpenAI  # Update import

# Import global configuration and local preprocessing module
import global_options as gl
from preprocess_earningscall import NlpPreProcess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bertopic_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_cuda() -> str:
    """
    Initializes CUDA if available and returns the device string.

    Returns:
        str: "cuda:X" if GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
        device = f'cuda:{torch.cuda.current_device()}'
        gpu_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        logger.info(f"Using GPU: {gpu_props.name}")
        logger.info(f"GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB")
        return device
    else:
        logger.warning("CUDA not available, using CPU")
        return "cpu"


class DataHandler:
    """
    Handles data loading and preprocessing operations.
    """
    def __init__(self, file_path: str, year_start: int, year_end: int):
        self.file_path = file_path
        self.year_start = year_start
        self.year_end = year_end

    def load_data(self) -> pd.DataFrame:
        """
        Loads text data from a CSV file in chunks and filters rows by year.

        Returns:
            pd.DataFrame: The concatenated DataFrame with filtered rows.
        """
        logger.info(f"Loading data for years {self.year_start}-{self.year_end}")
        try:
            # Read header to determine expected columns
            df_header = pd.read_csv(self.file_path, nrows=0)
            expected_cols = len(df_header.columns)
            logger.info(f"Expected columns: {expected_cols}\nColumns: {df_header.columns.tolist()}")

            chunks = pd.read_csv(
                self.file_path,
                chunksize=gl.CHUNK_SIZE,
                quotechar='"',
                doublequote=True,
                encoding='utf-8',
                engine='c',
                on_bad_lines='warn',
                delimiter=',',
                quoting=1
            )
            meta = pd.DataFrame()
            total_rows = 0
            estimated_rows = os.path.getsize(self.file_path) // 500  # rough estimate

            with tqdm(total=estimated_rows, desc="Loading data", ncols=100, colour="green") as pbar:
                for chunk in chunks:
                    if len(chunk.columns) != expected_cols:
                        logger.warning(f"Found {len(chunk.columns)} columns, expected {expected_cols}")
                        continue
                    # Process and filter the date-related column
                    chunk['year'] = pd.to_datetime(chunk['mostimportantdateutc'], errors='coerce').dt.year
                    filtered_chunk = chunk[(chunk['year'] >= self.year_start) &
                                             (chunk['year'] <= self.year_end)]
                    if not filtered_chunk.empty:
                        meta = pd.concat([meta, filtered_chunk], ignore_index=True)
                        total_rows += len(filtered_chunk)
                        if total_rows % (gl.CHUNK_SIZE * 10) == 0:
                            logger.info(f"Processed {total_rows} rows")
                    pbar.update(len(chunk))
            logger.info(f"Final dataset size: {len(meta)} rows")
            return meta

        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            logger.error(traceback.format_exc())
            raise

    def load_doc_parallel(self, path):
        logger.info(f"Loading documents from {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            
            num_processes = cpu_count() // 2
            chunk_size = max(1, num_lines // num_processes)
            chunks = []
            with open(path, 'r', encoding='utf-8') as f:
                current_chunk = []
                for i, line in enumerate(f):
                    current_chunk.append(line.strip())
                    if len(current_chunk) >= chunk_size and i < num_lines - 1:
                        chunks.append(current_chunk)
                        current_chunk = []
                if current_chunk:
                    chunks.append(current_chunk)
            
            with Pool(num_processes) as pool:
                results = list(tqdm(
                    pool.imap(self._process_chunk, chunks),
                    total=len(chunks),
                    desc="Processing document chunks"
                ))
            
            docs = [doc for chunk in results for doc in chunk if doc]
            docs = [doc.strip() for doc in docs if len(doc.strip()) > 30]
            logger.info(f"Loaded {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise        

    def preprocess_text(self, data: pd.DataFrame) -> list:
        """
        Preprocesses text data by filtering, cleaning, and removing duplicates.

        Args:
            data (pd.DataFrame): Raw data DataFrame.

        Returns:
            list: List of processed document strings.
        """
        logger.info("Starting text preprocessing")
        try:
            # Filter rows
            data = data.query('speakertypeid != 1')
            logger.info(f"After speaker filter: {len(data)} rows")

            data['text'] = data[gl.TEXT_COLUMN].astype(str)
            data['post_date'] = pd.to_datetime(data[gl.DATE_COLUMN], format='%Y-%m-%d', errors='coerce')
            data['post_year'] = data['post_date'].dt.year
            data['post_quarter'] = data['post_date'].dt.month
            data['yearq'] = data['post_year'].astype(str) + 'Q' + data['post_quarter'].astype(str)

            if 'Unnamed: 0' in data.columns:
                data.drop(columns=['Unnamed: 0'], inplace=True)

            nlp = NlpPreProcess()
            processed_texts = []
            batch_size = 100
            for i in tqdm(range(0, len(data), batch_size), desc="Processing text"):
                batch = data.iloc[i:i + batch_size]
                batch_processed = batch.apply(lambda x: nlp.preprocess_file(pd.DataFrame([x]), 'text')[0],
                                              axis=1)
                processed_texts.extend(batch_processed)
            data['text'] = processed_texts
            data.drop_duplicates(subset='text', keep='first', inplace=True)
            docs = [str(text) for text in data['text'] if len(str(text)) > 30]
            logger.info(f"Final number of documents: {len(docs)}")
            return docs

        except Exception as e:
            logger.error(f"Error in preprocess_text: {e}")
            logger.error(traceback.format_exc())
            raise


class EmbeddingGenerator:
    """
    Generates embeddings for a list of documents using SentenceTransformer.
    """
    def __init__(self, device: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.device = device
        try:
            self.embedding_model = SentenceTransformer(model_name, device=device)
            self.embedding_model.max_seq_length = 512
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    def _calculate_optimal_batch_size(self, embedding_dim: int, default_batch_size: int = gl.BATCH_SIZE) -> int:
        """
        Determines an optimal batch size based on the GPU memory.

        Args:
            embedding_dim (int): Dimension of the embedding vector.
            default_batch_size (int): Default fallback batch size.

        Returns:
            int: Calculated optimal batch size.
        """
        if "cuda" in self.device.lower():
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory
                mem_per_doc = embedding_dim * 4  # 4 bytes for float32
                optimal_batch = int((total_memory * 0.7) / mem_per_doc)
                optimal_batch = min(optimal_batch, 512)
                logger.info(f"Optimal batch size based on GPU memory: {optimal_batch}")
                return optimal_batch
            except Exception as e:
                logger.error(f"Error retrieving GPU properties: {e}, falling back to default batch size.")
                return default_batch_size
        else:
            logger.warning("CUDA not available. Using default batch size for CPU processing.")
            return default_batch_size

    def generate_embeddings(self, docs: list) -> np.ndarray:
        """
        Generates embeddings for documents in batches.

        Args:
            docs (list): List of document strings.

        Returns:
            np.ndarray: Array of document embeddings.
        """
        start_time = time.time()
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        batch_size = self._calculate_optimal_batch_size(embedding_dim, default_batch_size=gl.BATCH_SIZE)
        embeddings = []
        logger.info(f"Generating embeddings for {len(docs)} documents with batch size {batch_size}")
        for i in tqdm(range(0, len(docs), batch_size), desc="Generating embeddings"):
            batch = docs[i:i + batch_size]
            if "cuda" in self.device.lower():
                with torch.amp.autocast(device_type='cuda'):
                    batch_embed = self.embedding_model.encode(batch, device=self.device,
                                                              show_progress_bar=False,
                                                              normalize_embeddings=True)
            else:
                batch_embed = self.embedding_model.encode(batch, device=self.device,
                                                          show_progress_bar=False,
                                                          normalize_embeddings=True)
            embeddings.append(batch_embed)
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings shape: {embeddings.shape} in {time.time() - start_time:.2f} seconds")
        return embeddings


class TopicModeler:
    """
    Wraps topic modeling components (UMAP, HDBSCAN, BERTopic) and handles training
    and saving of results.
    """
    def __init__(self, device: str):
        self.device = device
        self.umap_model = UMAP(
            n_neighbors=gl.N_NEIGHBORS[0],
            n_components=gl.N_COMPONENTS[0],
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            verbose=True,
            low_memory=False,
            n_jobs=-1
        )
        self.hdbscan_model = HDBSCAN(
            min_samples=gl.MIN_SAMPLES[0],
            min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        self.representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3),
            "POS": PartOfSpeech("en_core_web_sm")
        }
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=None,  # We pass embeddings directly later
            min_topic_size=gl.MIN_CLUSTER_SIZE[0],
            calculate_probabilities=False,
            verbose=True,
            vectorizer_model=TfidfVectorizer(
                max_df=gl.MAX_DF[0],
                min_df=gl.MIN_DF[0],
                stop_words='english',
                ngram_range=(1, 3),
                use_idf=True,
                smooth_idf=True
            ),
            representation_model=self.representation_model
        )

        # Initialize OpenAI client
        try:
            api_key_path = os.path.join(os.getcwd(), 'data', 'OPENAI_API_KEY.txt')
            with open(api_key_path, 'r') as f:
                self.client = OpenAI(api_key=f.read().strip())
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise

    def train_topic_model(self, docs: list, embeddings: np.ndarray) -> BERTopic:
        """
        Fits the BERTopic model using the provided documents and embeddings.

        Args:
            docs (list): List of document strings.
            embeddings (np.ndarray): Corresponding document embeddings.

        Returns:
            BERTopic: Trained topic model.
        """
        start_time = time.time()
        logger.info(f"Clustering {len(docs)} documents for topic modeling")
        self.topic_model.fit(docs, embeddings)
        logger.info(f"Topic modeling completed in {time.time() - start_time:.2f} seconds")
        return self.topic_model

    def generate_topic_label(self, keywords: list, docs: list = None) -> str:
        """Generate descriptive label for topic using GPT."""
        try:
            # Format keywords and sample docs
            top_keywords = [k.replace('_', ' ') for k in keywords[:5]]
            doc_context = ""
            if docs:
                doc_context = "\nExample discussions:\n" + "\n".join(docs[:2])

            prompt = f"""
            Create a concise business topic label (2-4 words) for these earnings call keywords:
            
            Primary Keywords: {', '.join(top_keywords)}
            Secondary Keywords: {', '.join([k.replace('_', ' ') for k in keywords[5:8]])}
            {doc_context}

            Requirements:
            - Use standard financial/business terminology
            - Be specific but concise
            - Focus on the main business concept
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call analysis. Please read the following keywords representatives and name the topic in a concise manner."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=30
            )
            
            label = response.choices[0].message['content'].strip()
            return label

        except Exception as e:
            logger.error(f"Error generating topic label: {e}")
            return ' & '.join(k.replace('_', ' ').title() for k in keywords[:3])

    def save_topic_keywords(self, topic_model: BERTopic) -> None:
        """
        Saves topic keywords. If the 'Name' column is missing in topic_info,
        it is created from the topics dictionary.

        Args:
            topic_model (BERTopic): Trained topic model.
        """
        try:
            topic_info = topic_model.get_topic_info()
            topics_dict = topic_model.get_topics()
            
            # Generate labels for each topic
            labels = []
            for _, row in tqdm(topic_info.iterrows(), desc="Generating topic labels"):
                if row['Topic'] == -1:
                    labels.append("No Topic")
                    continue
                    
                keywords = [word for word, _ in topics_dict.get(row['Topic'], [])]
                docs = row.get('Representative_Docs', [])
                label = self.generate_topic_label(keywords, docs)
                labels.append(label)
            
            topic_info['Label'] = labels
            
            # Save results with correct variable names
            output_path = os.path.join(
                gl.output_folder, 
                f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_" \
                f"{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_" \
                f"{gl.YEAR_START}_{gl.YEAR_END}.csv"  # Changed from START_YEAR and YEAR_FILTER
            )
            topic_info.to_csv(output_path, index=False)
            logger.info(f"Saved topic keywords with labels to {output_path}")

        except Exception as e:
            logger.error(f"Error in save_topic_keywords: {e}")
            logger.error(traceback.format_exc())

    def save_figures(self, topic_model: BERTopic) -> None:
        """
        Saves various visualization figures generated by the topic model.
        
        Args:
            topic_model (BERTopic): Trained topic model.
        """
        try:
            visualization_path = os.path.join(gl.output_fig_folder, f'bertopic{gl.num_topic_to_plot}.pdf')
            fig = topic_model.visualize_barchart(top_n_topics=gl.num_topic_to_plot)
            fig.write_image(visualization_path)
            fig1 = topic_model.visualize_topics()
            fig1.write_image(visualization_path.replace('.pdf', f'_intertopic_distance_map_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.YEAR_START}_{gl.YEAR_END}.pdf'))
            fig2 = topic_model.visualize_heatmap()
            fig2.write_image(visualization_path.replace('.pdf', f'_heatmap_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.YEAR_START}_{gl.YEAR_END}.pdf'))
            fig3 = topic_model.visualize_hierarchy()
            fig3.write_image(visualization_path.replace('.pdf', f'_hierarchy_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.YEAR_START}_{gl.YEAR_END}.pdf'))
            logger.info(f"Visualization saved to {visualization_path}")
        except Exception as e:
            logger.error(f"Error in save_figures: {e}")
            logger.error(traceback.format_exc())


def main() -> None:
    """
    Main processing pipeline:
      1. Setup environment (CUDA/device)
      2. Load and preprocess data.
      3. Generate document embeddings.
      4. Train the topic model.
      5. Save topic keywords and visualization figures.
    """
    try:
        device = setup_cuda()
        file_path = os.path.join(os.getcwd(), 'data', gl.data_filename)
        data_handler = DataHandler(file_path, gl.YEAR_START, gl.YEAR_END)

        # Check if the preprocessed docs already exist
        docs_path = os.path.join(gl.output_folder, f'preprocessed_docs_{gl.YEAR_START}_{gl.YEAR_END}.txt')
        if os.path.exists(docs_path):
            logger.info(f"Found preprocessed docs at {docs_path}. Loading...")
            docs = data_handler.load_doc_parallel(docs_path)  # Use class method
        else:
            logger.info("Processed docs not found. Processing raw data...")
            data = data_handler.load_data()
            docs = data_handler.preprocess_text(data)
            os.makedirs(gl.output_folder, exist_ok=True)
            logger.info(f"Saving processed docs to {docs_path}")
            with open(docs_path, 'w', encoding='utf-8') as f:
                for doc in docs:
                    f.write(doc + "\n")

        # Generate embeddings
        embedding_gen = EmbeddingGenerator(device)
        embeddings = embedding_gen.generate_embeddings(docs)

        # Train topic model
        topic_modeler = TopicModeler(device)
        topic_model = topic_modeler.train_topic_model(docs, embeddings)

        # Save results
        topic_modeler.save_topic_keywords(topic_model)
        topic_modeler.save_figures(topic_model)

        # write Final log = "Topic modeling completed successfully"
        logger.info("Topic modeling completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 