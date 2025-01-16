import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import global_options as gl
from preprocess_earningscall import NlpPreProcess
import warnings
from sentence_transformers import SentenceTransformer
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from cuml.manifold import UMAP
# from cuml.cluster import HDBSCAN
from hdbscan import HDBSCAN
from umap import UMAP  # Import from umap-learn, not cuml
from bertopic import BERTopic
from sklearn.cluster import MiniBatchKMeans
# from model_selection_hpc import vectorize_doc
import torch
torch.cuda.empty_cache()
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from visualize_topic_models import VisualizeTopics as vt
import itertools
from matplotlib import pyplot as plt
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, TextGeneration
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import PCA
import joblib
from joblib import Memory
print(Memory)
warnings.filterwarnings('ignore')
current_path = os.getcwd()
tqdm.pandas()
joblib.Parallel(n_jobs=1)
import openai
import time

class BERTopicGPU(object):
    def __init__(self):
        # Increase batch size and optimize CUDA memory usage
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        
        # Initialize the embedding model with larger batch size
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        self.embedding_model.max_seq_length = 512  # Increase max sequence length if needed
        
        # Optimize UMAP for GPU memory usage
        self.umap_model = UMAP(
            n_components=gl.N_COMPONENTS[0],
            n_neighbors=gl.N_NEIGHBORS[0],
            random_state=42,
            metric=gl.METRIC[0],
            verbose=True,
            low_memory=False,  # Changed to False to use more memory but faster processing
            n_jobs=-1,
            transform_queue_size=4  # Increase queue size for parallel processing
        )
        
        # Optimize HDBSCAN for better performance
        self.hdbscan_model = HDBSCAN(
            min_samples=gl.MIN_SAMPLES[0],
            min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
            prediction_data=True,
            core_dist_n_jobs=-1,  # Use all CPU cores
            algorithm='best',
            memory=Memory(location=gl.output_folder)  # Cache computations
        )
        
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
                "POS": PartOfSpeech("en_core_web_sm"),
            }

        # Get API key from environment variable
        try:
            # Read API key from file
            api_key_path = os.path.join(os.getcwd(), 'data', 'OPENAI_API_KEY.txt')
            with open(api_key_path, 'r') as f:
                openai.api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"OpenAI API key file not found at {api_key_path}")
        except Exception as e:
            raise ValueError(f"Error setting OpenAI API key: {e}")
            
        # Set OpenAI API key
        self.file_path = os.path.join(current_path, 'data', 'earnings_calls_20231017.csv')


    def load_data(self):
        # Check if the file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file at path {self.file_path} does not exist.")

        # Define the file path and the number of rows to read as a subsample
        chunk_size = gl.CHUNK_SIZE  # Adjust this number to read a subsample
        # Use chunksize to limit rows number per iteration
        meta = pd.DataFrame()
        try:
            chunk_reader = pd.read_csv(
                                        self.file_path, 
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
        
        # Process in smaller sub-batches if needed
        sub_batch_size = 128
        batch_embeds = []
        
        for j in range(0, len(batch), sub_batch_size):
            sub_batch = batch[j:j + sub_batch_size]
            with torch.cuda.amp.autocast():  # Enable automatic mixed precision
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
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            os.system("nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv")
    
    def Bertopic_run(self, docs):
        print("Starting BERTopic processing...")
        
        # Calculate optimal batch size based on available GPU memory
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        # Estimate memory per document (in bytes)
        mem_per_doc = embedding_dim * 4  # 4 bytes per float32
        # Use 80% of available GPU memory
        optimal_batch_size = int((total_gpu_memory * 0.8) / mem_per_doc)
        batch_size = min(optimal_batch_size, 1024)  # Cap at 1024 for stability
        
        print(f"Using batch size: {batch_size}")
        
        # Initialize embeddings array with float32 instead of float64
        embeddings = np.zeros((len(docs), embedding_dim), dtype=np.float32)
        
        # Process documents in optimized batches
        for i in tqdm(range(0, len(docs), batch_size), colour="Blue"):
            batch_embed, i, i_end = self.process_batch_gpu(i, batch_size, docs, self.embedding_model, len(docs))
            embeddings[i:i_end, :] = batch_embed
            
            # Explicit GPU memory cleanup
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

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
        # Use in your code
        self.print_gpu_memory()  # Before UMAP
        # reduced_embeddings = self.reduce_dimensionality(embeddings, n_components=50)

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
            topic_model.fit_transform(docs, embeddings=embeddings)
        except ValueError as e:
            print(f"Error during BERTopic fitting: {e}")
            raise
        self.print_gpu_memory()  # After UMAP
        topic_model.save(os.path.join(gl.model_folder, f"bertopic_model_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.START_YEAR}_{gl.YEAR_FILTER}.pkl"))
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
        fig1.write_image(visualization_path.replace('.pdf', f'_intertopic_distance_map_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.START_YEAR}_{gl.YEAR_FILTER}.pdf'))
        fig2 = topic_model.visualize_heatmap()
        fig2.write_image(visualization_path.replace('.pdf', f'_heatmap_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.START_YEAR}_{gl.YEAR_FILTER}.pdf'))
        fig3 = topic_model.visualize_hierarchy()
        fig3.write_image(visualization_path.replace('.pdf', f'_hierarchy_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.START_YEAR}_{gl.YEAR_FILTER}.pdf'))
        print(f"Visualization saved to {visualization_path}")

    # def load_doc(self, path):
    #     # load the doc from a txt file to a list
    #     with open(path, 'r') as f:
    #         return f.readlines()
    def reduce_dimensionality(self, embeddings, n_components=50):
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
        return reduced_embeddings
        
    def load_doc_chunk(self, chunk_start, chunk_size, path):
        """Load a specific chunk of the document."""
        with open(path, 'r') as f:
            f.seek(chunk_start)  # Move to the start of the chunk
            lines = f.read(chunk_size).splitlines()
        return lines

    def chunkify_file(self, path, num_chunks=cpu_count()):
        """Determine file chunks for multiprocessing."""
        with open(path, 'r') as f:
            f.seek(0, 2)  # Move to the end of the file
            file_size = f.tell()
            chunk_size = file_size // num_chunks
        
        chunk_starts = [i * chunk_size for i in range(num_chunks)]
        return chunk_starts, chunk_size

    def load_doc_parallel(self, path):
        """Load the doc from a txt file to a list using multiple processes."""
        num_chunks = cpu_count()
        chunk_starts, chunk_size = self.chunkify_file(path, num_chunks)

        with Pool(num_chunks) as pool:
            docs = pool.starmap(self.load_doc_chunk, [(start, chunk_size, path) for start in chunk_starts])
        
        # Flatten the list of lists into a single list
        docs = [str(line).strip() for chunk in docs for line in chunk if line]
        # Filter out empty strings and ensure minimum length
        docs = [doc for doc in docs if len(doc) > 30]
        return docs

        
    def save_topic_keywords(self, topic_model):
        """Save topic information with additional theme column"""
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Add representative documents
        docs_per_topic = topic_model.get_representative_docs()
        topic_info['Representative_Docs'] = topic_info['Topic'].map(
            lambda x: docs_per_topic.get(x, [])
        )
        
        # Process themes in batches to avoid rate limits
        print("Generating themes for topics...")
        batch_size = 5  # Process 5 topics at a time
        themes = []
        
        for i in tqdm(range(0, len(topic_info), batch_size)):
            batch = topic_info.iloc[i:i+batch_size]
            batch_themes = []
            
            for _, row in batch.iterrows():
                theme = self.generate_topic_theme(
                    eval(row['Representation']) if isinstance(row['Representation'], str) else row['Representation'],
                    row['Representative_Docs']
                )
                theme = self.validate_theme(theme, eval(row['Representation']))
                batch_themes.append(theme)
                time.sleep(0.5)  # Rate limiting for API calls
                
            themes.extend(batch_themes)
        
        topic_info['Theme'] = themes
        
        # Save to CSV
        output_path = os.path.join(
            gl.output_folder, 
            f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{gl.NR_TOPICS[0]}_{gl.START_YEAR}_{gl.YEAR_FILTER}.csv"
        )
        topic_info.to_csv(output_path, index=False)
        print(f"Topic information saved to {output_path}")

    def create_text_generation_model(self):
        """Create a custom text generation model that uses OpenAI API"""
        class CustomTextGeneration:
            def transform(self, topic_words):
                # Convert topic words to a readable format
                words_str = ', '.join([word for word, _ in topic_words[:10]])
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a financial topic analyzer."},
                            {"role": "user", "content": f"Create a 2-4 word business theme based on these keywords: {words_str}"}
                        ],
                        temperature=0.3,
                        max_tokens=10
                    )
                    return response.choices[0].message['content'].strip()
                except Exception as e:
                    print(f"Error in text generation: {e}")
                    # Fallback: return concatenated top words
                    return ' '.join([word for word, _ in topic_words[:3]])

        return CustomTextGeneration()

    def generate_topic_theme(self, keywords, representative_docs):
        """Generate a descriptive theme using GPT for a set of keywords and representative documents"""
        try:
            # Clean and format keywords
            if isinstance(keywords, str):
                keywords = eval(keywords) if keywords.startswith('[') else keywords.split(', ')
            
            # Take top keywords and clean them
            top_keywords = [k.replace('_', ' ') for k in keywords[:5]]
            
            # Get representative text samples (limited to reduce token count)
            doc_samples = representative_docs[:2] if representative_docs else []
            doc_context = "\nExample discussions:\n" + "\n".join(doc_samples) if doc_samples else ""
            
            # Create a focused prompt
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
            
            Example good themes for different keyword sets:
            - revenue, growth, margin → "Revenue Growth Performance"
            - capacity, utilization, efficiency → "Operational Capacity Management"
            - market, share, penetration → "Market Share Expansion"
            - product, launch, innovation → "Product Innovation Strategy"
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call topic analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=30,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            theme = response.choices[0].message['content'].strip()
            
            # Validate theme length and format
            words = theme.split()
            if len(words) < 2 or len(words) > 5:
                # Fallback to a simpler format if theme is too long/short
                return ' '.join(word.title() for word in top_keywords[:3])
            
            return theme

        except Exception as e:
            print(f"Error generating theme: {e}")
            # Fallback: Create a simple theme from top keywords
            fallback_keywords = [k.replace('_', ' ').title() for k in keywords[:3]]
            return ' & '.join(fallback_keywords)

    def optimize_model_parameters(self, docs, test_params=None):
        """
        Optimize BERTopic model parameters using grid search and coherence scores.
        
        Args:
            docs: List of documents
            test_params: Dictionary of parameters to test (optional)
        
        Returns:
            best_params: Dictionary of optimal parameters
            best_score: Best coherence score achieved
        """
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
        
        # Calculate embeddings once to reuse
        print("Calculating document embeddings...")
        embeddings = np.zeros((len(docs), self.embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)
        batch_size = gl.BATCH_SIZE
        
        for i in tqdm(range(0, len(docs), batch_size), desc="Computing embeddings"):
            batch_embed, _, i_end = self.process_batch_gpu(i, batch_size, docs, self.embedding_model, len(docs))
            embeddings[i:i_end, :] = batch_embed
        
        # Generate parameter combinations
        param_combinations = [dict(zip(test_params.keys(), v)) 
                             for v in itertools.product(*test_params.values())]
        
        for params in tqdm(param_combinations, desc="Testing parameter combinations"):
            try:
                # Update models with current parameters
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
                
                # Create and fit topic model
                topic_model = BERTopic(
                    embedding_model=self.embedding_model,
                    umap_model=self.umap_model,
                    hdbscan_model=self.hdbscan_model,
                    vectorizer_model=self.vectorizer,
                    calculate_probabilities=True,
                    verbose=True
                )
                
                # Fit the model
                topic_model.fit_transform(docs, embeddings=embeddings)
                
                # Calculate coherence score
                coherence_score = self.compute_coherence_score(topic_model, docs)
                
                # Store results
                results.append({
                    'params': params,
                    'coherence_score': coherence_score,
                    'n_topics': len(topic_model.get_topics())
                })
                
                # Update best parameters if necessary
                if coherence_score > best_score:
                    best_score = coherence_score
                    best_params = params
                    
                # Clear GPU memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(gl.output_folder, 'parameter_optimization_results.csv'), index=False)
        
        print(f"\nBest parameters found:")
        print(f"Parameters: {best_params}")
        print(f"Coherence score: {best_score}")
        
        return best_params, best_score

    def generate_topic_name(self, keywords):
        """Generate a descriptive topic name from a list of keywords"""
        # Convert string representation of list to actual list
        if isinstance(keywords, str):
            # Remove brackets and quotes, split by commas
            keywords = keywords.strip("[]'").replace("'", "").split(", ")
        
        # Dictionary mapping keyword patterns to topic names
        topic_patterns = {
            # Financial Performance & Metrics
            ('growth', 'outlook', 'seasonality'): 'Growth and Financial Outlook',
            ('gross_margin', 'margin', 'revenue'): 'Margin and Revenue Performance',
            ('tax_rate', 'tax_reform', 'tax'): 'Tax and Regulatory Matters',
            ('share_buyback', 'share_repurchase', 'capital'): 'Capital Allocation and Buybacks',
            
            # Operations & Efficiency
            ('capacity_utilization', 'efficiency', 'automation'): 'Operational Efficiency',
            ('product', 'launch', 'innovation'): 'Product Development and Innovation',
            ('market', 'share', 'penetration'): 'Market Position and Strategy',
            
            # Corporate Communications
            ('prepared_remark', 'prepare_remark', 'mention_prepared'): 'Prepared Remarks',
            ('organic_growth', 'organic', 'dry_powder'): 'Organic Growth Strategy',
            ('geography', 'region', 'expansion'): 'Geographic Expansion',
            ('mix', 'product', 'portfolio'): 'Product Mix and Portfolio',
            
            # Management & Structure
            ('board', 'restructuring', 'change'): 'Corporate Governance',
            ('give_color', 'timing', 'time_frame'): 'Forward-Looking Commentary',
            ('permit', 'regulatory', 'compliance'): 'Regulatory Compliance',
            ('headwind', 'tailwind', 'margin'): 'Business Environment Factors',
            
            # Operations & Logistics
            ('turnover', 'retention', 'talent'): 'Workforce Management',
            ('supply_chain', 'supplier', 'supply'): 'Supply Chain Management',
            ('store', 'online', 'sale'): 'Retail and E-commerce',
            ('watch_list', 'stay_tune', 'tenant'): 'Risk Monitoring',
            ('dry_dock', 'container', 'port'): 'Maritime Operations'
        }
        
        # Check keywords against patterns
        for pattern, topic_name in topic_patterns.items():
            if any(keyword in keywords for keyword in pattern):
                return topic_name
            
        # Default case - take first 3 keywords and make a generic name
        first_three = keywords[:3]
        return ' & '.join(word.replace('_', ' ').title() for word in first_three)

    def add_topic_names(self, df):
        """Add descriptive topic names to the dataframe"""
        df['Topic_Name'] = df['Representation'].apply(self.generate_topic_name)
        return df

    def validate_theme(self, theme, keywords):
        """Validate and clean generated themes"""
        # Remove any unwanted characters or formatting
        theme = theme.strip('"\'').strip()
        
        # Check if theme is too generic
        generic_terms = {'topic', 'theme', 'discussion', 'earnings call', 'business'}
        theme_words = set(theme.lower().split())
        
        if theme_words.issubset(generic_terms):
            # If theme is too generic, use keywords
            return ' '.join(k.replace('_', ' ').title() for k in keywords[:3])
        
        # Ensure proper capitalization
        theme = ' '.join(word.capitalize() for word in theme.split())
        
        return theme

    
if __name__ == "__main__":
    bt = BERTopicGPU()
    docs_path = os.path.join(gl.output_folder, f'preprocessed_docs_{gl.START_YEAR}_{gl.YEAR_FILTER}.txt')
    
    if os.path.exists(docs_path):
        print("Reading preprocessed docs from preprocessed_docs.txt")
        docs = bt.load_doc_parallel(docs_path)
        docs = list(set(docs))
    else:
        meta = bt.load_data()
        docs = bt.pre_process_text(meta)
        bt.save_file(docs, docs_path, bar_length=100)

    # Skip optimization and use predefined parameters
    topic_model = bt.Bertopic_run(docs)
    bt.save_topic_keywords(topic_model)
    bt.save_figures(topic_model)
    print("BERTopic model training completed.")



'''
Compare the updates with the v3.py
 # Old parameters                # New parameters
   N_NEIGHBORS = [28]             → [15]      # Less memory, still effective
   N_COMPONENTS = [6]             → [5]       # Simpler dimensionality
   MIN_DIST = [0.0]              → [0.1]     # Better cluster separation
   MIN_SAMPLES = [15]            → [10]      # More granular topics
   MIN_CLUSTER_SIZE = [40]       → [30]      # Smaller but meaningful clusters
   NR_TOPICS = [150]             → [100]     # More focused topic count
   


  # Old workflow
   1. Load/preprocess docs
   2. Run BERTopic with fixed parameters
   3. Save results

   # New workflow
   1. Load/preprocess docs
   2. Run parameter optimization
   3. Run BERTopic with optimized parameters
   4. Save results

graph TD
    A[Start] --> B[Initialize BERTopicGPU]
    B --> C{Check if preprocessed docs exist}
    
    C -->|Yes| D[Load preprocessed docs]
    C -->|No| E[Load raw data & preprocess]
    E --> F[Save preprocessed docs]
    
    D --> G[Parameter Optimization Phase]
    F --> G
    
    G --> H[Calculate embeddings once]
    H --> I[Test different parameter combinations]
    I --> J[Calculate coherence scores]
    J --> K[Find best parameters]
    K --> L[Save optimization results]
    
    L --> M[Update UMAP & HDBSCAN with best params]
    M --> N[Train final BERTopic model]
    N --> O[Save topic keywords]
    O --> P[Save visualizations]
    P --> Q[End]

    subgraph "Parameter Optimization Loop"
        I --> J
        J --> I
    end
'''