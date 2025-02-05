import os
import sys
import time
import warnings
import logging
import traceback
from multiprocessing import Pool, cpu_count

# Import third-party libraries
import itertools
import torch
import numpy as np
import pandas as pd
import openai
import dask.dataframe as dd
import swifter
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic import BERTopic
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from matplotlib import pyplot as plt
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, TextGeneration
from sklearn.decomposition import PCA
import joblib
from joblib import Memory
import re

# Import local modules
import global_options as gl
from preprocess_earningscall import NlpPreProcess

def setup_cuda():
    """Initialize CUDA settings if available"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
        return True
    return False

# Configure basic settings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')
current_path = os.getcwd()
tqdm.pandas()
joblib.Parallel(n_jobs=1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bertopic_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize CUDA
has_cuda = setup_cuda()
if has_cuda:
    logger.info("CUDA initialized successfully")
else:
    logger.warning("CUDA not available, using CPU")

class BERTopicGPU(object):
    def __init__(self, YEAR_START, YEAR_END):
        self.YEAR_START = YEAR_START
        self.YEAR_END = YEAR_END
        
        # Initialize CUDA with proper error handling
        try:
            if torch.cuda.is_available():
                # Clear GPU memory first
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                
                # Get the first available GPU
                current_device = torch.cuda.current_device()
                self.device = f'cuda:{current_device}'
                
                # Log GPU info
                gpu_properties = torch.cuda.get_device_properties(current_device)
                logger.info(f"Using GPU: {gpu_properties.name}")
                logger.info(f"GPU Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
            else:
                self.device = 'cpu'
                logger.info("No GPU available, using CPU")
        except Exception as e:
            logger.warning(f"Error initializing GPU: {e}")
            self.device = 'cpu'
            logger.info("Falling back to CPU")
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
            self.embedding_model.max_seq_length = 512
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
        
        # Initialize UMAP with correct parameters
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
        
        # Initialize HDBSCAN with correct parameters
        self.hdbscan_model = HDBSCAN(
            min_samples=gl.MIN_SAMPLES[0],
            min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize BERTopic with correct parameters
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=self.embedding_model,
            min_topic_size=gl.MIN_CLUSTER_SIZE[0],
            calculate_probabilities=False,
            verbose=True
        )
        
        # Initialize TfidfVectorizer with desired parameters
        self.vectorizer = TfidfVectorizer(
            max_df=gl.MAX_DF[0],
            min_df=gl.MIN_DF[0],
            stop_words='english',
            ngram_range=(1, 1),
            use_idf=True,
            smooth_idf=True
        )
        
        self.representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3),
            "POS": PartOfSpeech("en_core_web_sm"),
        }

        # Get API key from environment variable
        try:
            api_key_path = os.path.join(os.getcwd(), 'data', 'OPENAI_API_KEY.txt')
            with open(api_key_path, 'r') as f:
                openai.api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"OpenAI API key file not found at {api_key_path}")
        except Exception as e:
            raise ValueError(f"Error setting OpenAI API key: {e}")
            
        # Set OpenAI API key
        self.file_path = os.path.join(current_path, 'data', gl.data_filename)

    def load_data(self):
        logger.info(f"Starting data loading for years {self.YEAR_START}-{self.YEAR_END}")
        
        try:
            df_header = pd.read_csv(self.file_path, nrows=0)
            file_size = os.path.getsize(self.file_path)
            estimated_rows = file_size // 500  # Rough estimate based on average row size
            
            expected_cols = len(df_header.columns)
            logger.info(f"Expected columns: {expected_cols}")
            logger.info(f"Columns: {df_header.columns.tolist()}")
            
            chunks = pd.read_csv(
                self.file_path,
                chunksize=gl.CHUNK_SIZE,
                quotechar='"',
                doublequote=True,
                escapechar=None,
                encoding='utf-8',
                engine='c',
                on_bad_lines='warn',
                delimiter=',',
                quoting=1
            )
            
            meta = pd.DataFrame()
            total_rows = 0
            
            with tqdm(total=estimated_rows, desc="Loading data", 
                      bar_format="{l_bar}{bar} [time left: {remaining}]", 
                      ncols=100, colour="green") as pbar:
                for chunk in chunks:
                    try:
                        if len(chunk.columns) != expected_cols:
                            logger.warning(f"Found {len(chunk.columns)} columns, expected {expected_cols}")
                            continue
                        
                        chunk['year'] = pd.to_datetime(
                            chunk['mostimportantdateutc'],
                            errors='coerce'
                        ).dt.year
                        
                        filtered_chunk = chunk[
                            (chunk['year'] >= self.YEAR_START) & 
                            (chunk['year'] <= self.YEAR_END)
                        ]
                        
                        if not filtered_chunk.empty:
                            meta = pd.concat([meta, filtered_chunk], ignore_index=True)
                            total_rows += len(filtered_chunk)
                            
                            if total_rows % (gl.CHUNK_SIZE * 10) == 0:
                                logger.info(f"Processed {total_rows} rows")
                        
                        pbar.update(len(chunk))
                    
                    except Exception as e:
                        logger.warning(f"Error in chunk: {str(e)}")
                        continue
            
            logger.info(f"Final dataset size: {len(meta)} rows")
            return meta

        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            logger.error(traceback.format_exc())
            raise

    def pre_process_text(self, data):
        logger.info("Starting text preprocessing")
        try:
            # Disable GPU for preprocessing
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
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
            batch_size = 100
            processed_texts = []
            
            for i in tqdm(range(0, len(data), batch_size), desc="Processing text"):
                batch = data.iloc[i:i+batch_size]
                batch_processed = batch.apply(
                    lambda x: nlp.preprocess_file(pd.DataFrame([x]), 'text')[0],
                    axis=1
                )
                processed_texts.extend(batch_processed)
            
            data['text'] = processed_texts
            data.drop_duplicates(subset='text', keep='first', inplace=True)
            docs = [str(text) for text in data['text'] if len(str(text)) > 30]
            
            logger.info(f"Final number of documents: {len(docs)}")
            
            # Re-enable GPU for later use (if desired)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return docs

        except Exception as e:
            logger.error(f"Error in pre_process_text: {e}")
            logger.error(traceback.format_exc())
            raise

    def filter_empty_topics(self, topics):
        filtered_topics = {}
        for topic_num, topic_words in topics.items():
            valid_words = [(word, score) for word, score in topic_words if word]
            if valid_words:
                filtered_topics[topic_num] = valid_words
        return filtered_topics

    def compute_coherence_score(self, topic_model, texts):
        topics = topic_model.get_topics()
        logger.info(f"Number of topics: {len(topics)}")
        filtered_topics = self.filter_empty_topics(topics)
        topics_list = [[word for word, _ in topic_words] 
                       for topic_num, topic_words in filtered_topics.items() if topic_num != -1]

        if isinstance(texts[0], str):
            texts = [doc.split() for doc in texts]

        dictionary = Dictionary(texts)
        coherence_model = CoherenceModel(
            topics=topics_list,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        return coherence_score

    def process_batch_gpu(self, i, batch_size, docs, embedding_model, N_):
        i_end = min(i + batch_size, N_)
        batch = docs[i:i_end]
        sub_batch_size = 128
        batch_embeds = []
        for j in range(0, len(batch), sub_batch_size):
            sub_batch = batch[j:j + sub_batch_size]
            with torch.amp.autocast(device_type='cuda'):
                sub_batch_embed = embedding_model.encode(
                    sub_batch,
                    device=self.device,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            batch_embeds.append(sub_batch_embed)
        batch_embed = np.vstack(batch_embeds)
        return batch_embed, i, i_end

    def print_gpu_memory(self):
        try:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(current_device)/1e9:.2f} GB")
                logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(current_device)/1e9:.2f} GB")
            else:
                logger.info("No GPU available")
        except Exception as e:
            logger.warning(f"Error getting GPU memory info: {e}")
    
    def _calculate_optimal_batch_size(self, embedding_dim, default_batch_size=128):
        if torch.cuda.is_available():
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory  # in bytes
                logger.info(f"GPU detected: {gpu_props.name} with {total_memory / 1e9:.2f}GB total memory")
                mem_per_doc = embedding_dim * 4  # assuming float32
                optimal_batch = int((total_memory * 0.7) / mem_per_doc)
                optimal_batch = min(optimal_batch, 512)
                logger.info(f"Optimal batch size based on GPU memory: {optimal_batch}")
                return optimal_batch
            except Exception as e:
                logger.error(f"Error retrieving GPU properties: {e}. Falling back to default batch size.")
                return default_batch_size
        else:
            logger.warning("CUDA not available. Using default batch size for CPU processing.")
            return default_batch_size

    def Bertopic_run(self, docs):
        if not docs:
            raise ValueError("Empty document list")
        
        start_time = time.time()
        logger.info(f"Starting BERTopic processing with {len(docs)} documents")
        if torch.cuda.is_available():
            device = "cuda"
            batch_size = 128
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            # return error
            raise ValueError("CUDA not available")
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()  # e.g., 384
        batch_size = self._calculate_optimal_batch_size(embedding_dim, default_batch_size=gl.BATCH_SIZE)
        
        embeddings = []
        for i in tqdm(range(0, len(docs), batch_size), desc="Processing embeddings"):
            batch = docs[i:min(i + batch_size, len(docs))]
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda'):
                    batch_embed = self.embedding_model.encode(
                        batch,
                        device=self.device,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
            else:
                batch_embed = self.embedding_model.encode(
                    batch,
                    device=self.device,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            embeddings.append(batch_embed)
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        embeddings = np.vstack(embeddings)
        
        if embeddings.shape[1] > 64:
            embeddings = self.reduce_dimensionality(embeddings, n_components=64)
        
        topic_model = self._train_topic_model(docs, embeddings)
        logger.info(f"Processing completed in {time.time() - start_time:.2f}s")
        return topic_model

    def _train_topic_model(self, docs, embeddings):
        return BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=self.embedding_model,
            min_topic_size=gl.MIN_CLUSTER_SIZE[0],
            calculate_probabilities=False,
            verbose=True
        ).fit(docs, embeddings)

    def save_file(self, data, path, bar_length=100):
        with open(path, 'w') as f:
            with tqdm(total=len(data), desc="Saving data", 
                      bar_format="{l_bar}{bar} [time left: {remaining}]", 
                      ncols=bar_length, colour="green") as pbar:
                for item in data:
                    f.write("%s\n" % item)
                    pbar.update(1)
                    
    def save_figures(self, topic_model):
        visualization_path = os.path.join(gl.output_fig_folder, f'bertopic{gl.num_topic_to_plot}.pdf')
        fig = topic_model.visualize_barchart(top_n_topics=gl.num_topic_to_plot)
        fig.write_image(visualization_path)
        fig1 = topic_model.visualize_topics()
        fig1.write_image(visualization_path.replace('.pdf', f'_intertopic_distance_map_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{self.YEAR_START}_{self.YEAR_END}.pdf'))
        fig2 = topic_model.visualize_heatmap()
        fig2.write_image(visualization_path.replace('.pdf', f'_heatmap_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{self.YEAR_START}_{self.YEAR_END}.pdf'))
        fig3 = topic_model.visualize_hierarchy()
        fig3.write_image(visualization_path.replace('.pdf', f'_hierarchy_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{self.YEAR_START}_{self.YEAR_END}.pdf'))
        logger.info(f"Visualization saved to {visualization_path}")

    def reduce_dimensionality(self, embeddings, n_components=50):
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
        return reduced_embeddings

    def manage_gpu_state(self, enable=True):
        """Manage GPU state in a flexible way"""
        try:
            if enable and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                devices = list(range(device_count))
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
                torch.cuda.empty_cache()
                logger.info(f"Enabled {len(devices)} GPU device(s)")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                logger.info("Disabled GPU devices")
        except Exception as e:
            logger.warning(f"Error managing GPU state: {e}")

    def load_doc_chunk(self, chunk_start, chunk_size, path):
        with open(path, 'r') as f:
            f.seek(chunk_start)
            lines = f.read(chunk_size).splitlines()
        return lines

    def chunkify_file(self, path, num_chunks=cpu_count()):
        with open(path, 'r') as f:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = file_size // num_chunks
        chunk_starts = [i * chunk_size for i in range(num_chunks)]
        return chunk_starts, chunk_size

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

    def _process_chunk(self, chunk):
        try:
            processed = []
            for line in chunk:
                if isinstance(line, str) and line.strip():
                    clean_line = line.strip()
                    clean_line = re.sub(r'\s+', ' ', clean_line)
                    if len(clean_line) > 30:
                        processed.append(clean_line)
            return processed
        except Exception as e:
            logger.warning(f"Error processing chunk: {e}")
            return []

    def _save_intermediate_results(self, topic_info_chunk, themes):
        path = os.path.join(gl.output_folder, 'intermediate_results.csv')
        topic_info_chunk['Theme'] = themes
        topic_info_chunk.to_csv(path, index=False)
        logger.info(f"Saved intermediate results to {path}")

    def _save_final_results(self, topic_info):
        path = os.path.join(gl.output_folder, 'final_topic_results.csv')
        topic_info.to_csv(path, index=False)
        logger.info(f"Saved final topic results to {path}")

    def save_topic_keywords(self, topic_model):
        try:
            topic_info = topic_model.get_topic_info()
            theme_cache = {}
            batch_size = 10
            themes = []
            logger.info(f"columns of topic_info: {topic_info.columns}")
            for i in tqdm(range(0, len(topic_info), batch_size), desc="Generating topic themes"):
                batch = topic_info.iloc[i:i+batch_size]
                batch_themes = []
                for _, row in batch.iterrows():
                    keywords = row['Representation']
                    cache_key = str(sorted(keywords[:5]))
                    if cache_key in theme_cache:
                        batch_themes.append(theme_cache[cache_key])
                        continue
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            theme = self.generate_topic_theme(keywords, row['Representative_Docs'])
                            theme_cache[cache_key] = theme
                            batch_themes.append(theme)
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to generate theme after {max_retries} attempts")
                                theme = ' & '.join(str(k).replace('_', ' ').title() for k in keywords[:3])
                                batch_themes.append(theme)
                        time.sleep(1)
                themes.extend(batch_themes)
                if i % (batch_size * 5) == 0:
                    self._save_intermediate_results(topic_info.iloc[:i+batch_size], themes)
            topic_info['Theme'] = themes
            self._save_final_results(topic_info)
        except Exception as e:
            logger.error(f"Error in save_topic_keywords: {e}")
            raise

    def generate_topic_theme(self, keywords, representative_docs):
        try:
            if isinstance(keywords, str):
                keywords = eval(keywords) if keywords.startswith('[') else keywords.split(', ')
            top_keywords = [k.replace('_', ' ') for k in keywords[:5]]
            doc_samples = representative_docs[:2] if representative_docs else []
            doc_context = "\nExample discussions:\n" + "\n".join(doc_samples) if doc_samples else ""
            prompt = f"""
            Analyze these earnings call keywords and create a concise business theme (2-4 words):
            
            Primary Keywords: {', '.join(top_keywords)}
            Secondary Keywords: {', '.join([k.replace('_', ' ') for k in keywords[5:8]])}
            {doc_context}

            Requirements:
            - Use standard financial/business terminology
            - Be specific but concise (2-3 words)
            - Focus on the main business concept or metric
            - Avoid generic terms like "business" or "corporate" unless essential
            """
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call topic analysis. Label the theme with input text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=30,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            theme = response.choices[0].message['content'].strip()
            words = theme.split()
            if len(words) < 2 or len(words) > 5:
                return ' '.join(word.title() for word in top_keywords[:3])
            return theme
        except Exception as e:
            logger.error(f"Error generating theme: {e}")
            fallback_keywords = [k.replace('_', ' ').title() for k in keywords[:3]]
            return ' & '.join(fallback_keywords)

    def optimize_model_parameters(self, docs, test_params=None):
        if test_params is None:
            test_params = {
                'n_neighbors': [5, 15, 30],
                'n_components': [3, 5, 7],
                'min_cluster_size': [20, 30, 40],
                'min_samples': [5, 10, 15]
            }
        
        best_score = -float('inf')
        best_params = None
        results = []
        
        logger.info("Calculating document embeddings for parameter optimization...")
        embeddings = np.zeros((len(docs), self.embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)
        batch_size = gl.BATCH_SIZE
        
        for i in tqdm(range(0, len(docs), batch_size), desc="Computing embeddings"):
            batch_embed, _, i_end = self.process_batch_gpu(i, batch_size, docs, self.embedding_model, len(docs))
            embeddings[i:i_end, :] = batch_embed
        
        param_combinations = [dict(zip(test_params.keys(), v)) 
                              for v in itertools.product(*test_params.values())]
        
        for params in tqdm(param_combinations, desc="Testing parameter combinations"):
            try:
                self.umap_model = UMAP(
                    n_neighbors=params['n_neighbors'],
                    n_components=params['n_components'],
                    random_state=42,
                    metric='cosine',
                    low_memory=False,
                    n_jobs=-1
                )
                self.hdbscan_model = HDBSCAN(
                    min_cluster_size=params['min_cluster_size'],
                    min_samples=params['min_samples'],
                    prediction_data=True,
                    core_dist_n_jobs=-1,
                    algorithm='best',
                    memory=Memory(location=gl.output_folder)
                )
                topic_model = BERTopic(
                    embedding_model=self.embedding_model,
                    umap_model=self.umap_model,
                    hdbscan_model=self.hdbscan_model,
                    vectorizer_model=self.vectorizer,
                    calculate_probabilities=False,
                    verbose=True
                )
                topic_model.fit_transform(docs, embeddings=embeddings)
                coherence_score = self.compute_coherence_score(topic_model, docs)
                results.append({
                    'params': params,
                    'coherence_score': coherence_score,
                    'n_topics': len(topic_model.get_topics())
                })
                if coherence_score > best_score:
                    best_score = coherence_score
                    best_params = params
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error with parameters {params}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(gl.output_folder, 'parameter_optimization_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Saved parameter optimization results to {results_csv_path}")
        logger.info(f"Best parameters found: {best_params} with coherence score: {best_score}")
        
        return best_params, best_score

    def generate_topic_name(self, keywords):
        if isinstance(keywords, str):
            keywords = keywords.strip("[]'").replace("'", "").split(", ")
        topic_patterns = {
            ('growth', 'outlook', 'seasonality'): 'Growth and Financial Outlook',
            ('gross_margin', 'margin', 'revenue'): 'Margin and Revenue Performance',
            ('tax_rate', 'tax_reform', 'tax'): 'Tax and Regulatory Matters',
            ('share_buyback', 'share_repurchase', 'capital'): 'Capital Allocation and Buybacks',
            ('capacity_utilization', 'efficiency', 'automation'): 'Operational Efficiency',
            ('product', 'launch', 'innovation'): 'Product Development and Innovation',
            ('market', 'share', 'penetration'): 'Market Position and Strategy',
            ('prepared_remark', 'prepare_remark', 'mention_prepared'): 'Prepared Remarks',
            ('organic_growth', 'organic', 'dry_powder'): 'Organic Growth Strategy',
            ('geography', 'region', 'expansion'): 'Geographic Expansion',
            ('mix', 'product', 'portfolio'): 'Product Mix and Portfolio',
            ('board', 'restructuring', 'change'): 'Corporate Governance',
            ('give_color', 'timing', 'time_frame'): 'Forward-Looking Commentary',
            ('permit', 'regulatory', 'compliance'): 'Regulatory Compliance',
            ('headwind', 'tailwind', 'margin'): 'Business Environment Factors',
            ('turnover', 'retention', 'talent'): 'Workforce Management',
            ('supply_chain', 'supplier', 'supply'): 'Supply Chain Management',
            ('store', 'online', 'sale'): 'Retail and E-commerce',
            ('watch_list', 'stay_tune', 'tenant'): 'Risk Monitoring',
            ('dry_dock', 'container', 'port'): 'Maritime Operations'
        }
        for pattern, topic_name in topic_patterns.items():
            if any(keyword in keywords for keyword in pattern):
                return topic_name
        first_three = keywords[:3]
        return ' & '.join(word.replace('_', ' ').title() for word in first_three)

    def add_topic_names(self, df):
        df['Topic_Name'] = df['Representation'].apply(self.generate_topic_name)
        return df

    def validate_theme(self, theme, keywords):
        if not isinstance(keywords, list):
            try:
                keywords = eval(keywords) if isinstance(keywords, str) else keywords
            except:
                keywords = keywords.strip("[]'").replace("'", "").split(", ")
        theme = theme.strip('"\'').strip()
        generic_terms = {'topic', 'theme', 'discussion', 'earnings call', 'business'}
        theme_words = set(theme.lower().split())
        if theme_words.issubset(generic_terms):
            return ' '.join(k.replace('_', ' ').title() for k in keywords[:3])
        theme = ' '.join(word.capitalize() for word in theme.split())
        return theme

    def monitor_memory(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1e9
                logger.info(f"GPU Memory: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Error monitoring memory: {e}")

    def initialize_gpu(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Set memory growth
            torch.cuda.set_per_process_memory_growth(True)
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
        return device


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    YEAR_START = 2005
    YEAR_END = 2010
    logger.info(f"Processing data for years {YEAR_START}-{YEAR_END}")
    
    # Disable GPU for preprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    bt = BERTopicGPU(YEAR_START, YEAR_END)
    docs_path = os.path.join(gl.output_folder, f'preprocessed_docs_{YEAR_START}_{YEAR_END}.txt')
    
    try:
        # Disable GPU explicitly for preprocessing using helper method
        bt.manage_gpu_state(enable=False)
        if os.path.exists(docs_path):
            logger.info(f"Loading preprocessed docs from {docs_path}")
            docs = bt.load_doc_parallel(docs_path)
            docs = list(set(docs))
            logger.info(f"Loaded {len(docs)} preprocessed documents")
        else:
            logger.info("Processing raw data...")
            meta = bt.load_data()
            logger.info(f"Loaded {len(meta)} rows of raw data")
            if not meta.empty:
                docs = bt.pre_process_text(meta)
                logger.info(f"Generated {len(docs)} documents")
                if docs:
                    bt.save_file(docs, docs_path, bar_length=100)
                    logger.info(f"Saved preprocessed docs to {docs_path}")
                else:
                    logger.error("No documents generated after preprocessing!")
                    sys.exit(1)
            else:
                logger.error("No data loaded!")
                sys.exit(1)
        
        # Re-enable GPU for BERTopic processing
        bt.manage_gpu_state(enable=True)
        torch.cuda.empty_cache()
        
        topic_model = bt.Bertopic_run(docs)
        bt.save_topic_keywords(topic_model)
        bt.save_figures(topic_model)
        logger.info("BERTopic model training completed.")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
