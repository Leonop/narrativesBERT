#!/usr/bin/env python

# Set environment variable for tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
BERTtopic_big_data_hpc_v7_refactored.py

Optimized pipeline:
  - Iterative chunk reading for lower memory footprint.
  - Parallel chunk processing with ProcessPoolExecutor.
  - Optimized GPU checks and dynamic batch sizing.
  
  Configuration and constants are assumed to be provided via the imported module 'global_options'.
"""

import sys
import time
import traceback
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
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
from openai import OpenAI  # Updated import

import global_options as gl
from preprocess_earningscall import NlpPreProcess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bertopic_processing.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_cuda() -> str:
    """Initialize CUDA if available and return device string."""
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
    """Handles data loading and preprocessing operations."""
    def __init__(self, file_path: str, year_start: int, year_end: int):
        self.file_path = file_path
        self.year_start = year_start
        self.year_end = year_end
        self.nlp_processor = NlpPreProcess()

    @staticmethod
    def _process_chunk(chunk: str) -> list:
        """Process a chunk of text into non-empty stripped lines."""
        try:
            return [line.strip() for line in chunk.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return []

    def load_doc_parallel(self, docs_path: str, chunk_size: int = 1024*1024) -> list:
        """Load documents from a file iteratively and process chunks in parallel."""
        docs = []
        logger.info(f"Loading documents from {docs_path}")

        def read_chunks(fp, size):
            while True:
                chunk = fp.read(size)
                if not chunk:
                    break
                yield chunk

        # Use ProcessPoolExecutor for parallel processing of chunks
        with open(docs_path, 'r', encoding='utf-8') as f, ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(DataHandler._process_chunk, chunk): chunk 
                       for chunk in read_chunks(f, chunk_size)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading documents"):
                docs.extend(future.result())
        logger.info(f"Loaded {len(docs)} documents")
        return docs

    def load_data(self) -> pd.DataFrame:
        """Load and filter CSV data by year."""
        logger.info(f"Loading data for years {self.year_start}-{self.year_end}")
        try:
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
            meta = []
            total_rows = 0
            estimated_rows = os.path.getsize(self.file_path) // 500  # rough estimate

            with tqdm(total=estimated_rows, desc="Loading data", ncols=100, colour="green") as pbar:
                for chunk in chunks:
                    if len(chunk.columns) != expected_cols:
                        logger.warning(f"Found {len(chunk.columns)} columns, expected {expected_cols}")
                        continue
                    chunk['year'] = pd.to_datetime(chunk['mostimportantdateutc'], errors='coerce').dt.year
                    filtered_chunk = chunk[(chunk['year'] >= self.year_start) & (chunk['year'] <= self.year_end)]
                    if not filtered_chunk.empty:
                        meta.append(filtered_chunk)
                        total_rows += len(filtered_chunk)
                        if total_rows % (gl.CHUNK_SIZE * 10) == 0:
                            logger.info(f"Processed {total_rows} rows")
                    pbar.update(len(chunk))
            df_meta = pd.concat(meta, ignore_index=True)
            logger.info(f"Final dataset size: {len(df_meta)} rows")
            return df_meta

        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            logger.error(traceback.format_exc())
            raise

    def preprocess_text(self, data: pd.DataFrame) -> list:
        """Preprocess text data by filtering, cleaning, and removing duplicates."""
        logger.info("Starting text preprocessing")
        try:
            data = data.query('speakertypeid != 1')
            logger.info(f"After speaker filter: {len(data)} rows")
            data['text'] = data[gl.TEXT_COLUMN].astype(str)
            data['post_date'] = pd.to_datetime(data[gl.DATE_COLUMN], format='%Y-%m-%d', errors='coerce')
            data['post_year'] = data['post_date'].dt.year
            data['post_quarter'] = data['post_date'].dt.month
            data['yearq'] = data['post_year'].astype(str) + 'Q' + data['post_quarter'].astype(str)
            data.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

            processed_texts = []
            batch_size = 100
            for i in tqdm(range(0, len(data), batch_size), desc="Processing text"):
                batch = data.iloc[i:i + batch_size]
                batch_processed = batch.apply(lambda x: self.nlp_processor.preprocess_file(pd.DataFrame([x]), 'text')[0],
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
    """Generates embeddings for documents using SentenceTransformer."""
    def __init__(self, device: str):
        self.device = device
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        self.base_batch_size = 512  # fallback batch size
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            # Correct GPU condition check
            if 'V100' in gpu_name and '32GB' in gpu_name:
                self.base_batch_size = 1536
            else:
                logger.warning("GPU memory unknown; using default batch size.")

    def _calculate_optimal_batch_size(self, embedding_dim: int, default_batch_size: int = gl.BATCH_SIZE) -> int:
        """Determine an optimal batch size based on GPU memory."""
        if "cuda" in self.device.lower():
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory
                mem_per_doc = embedding_dim * 4  # 4 bytes per float32
                optimal_batch = int((total_memory * 0.7) / mem_per_doc)
                optimal_batch = min(optimal_batch, 512)
                logger.info(f"Optimal batch size based on GPU memory: {optimal_batch}")
                return optimal_batch
            except Exception as e:
                logger.error(f"Error retrieving GPU properties: {e}, using default batch size.")
                return default_batch_size
        else:
            logger.warning("CUDA not available. Using default CPU batch size.")
            return default_batch_size

    def generate_embeddings(self, docs: list) -> np.ndarray:
        """Generate embeddings using a memory-mapped array for large datasets."""
        start_time = time.time()
        embedding_dim = self.model.get_sentence_embedding_dimension()
        batch_size = self._calculate_optimal_batch_size(embedding_dim)
        mmap_path = os.path.join(gl.output_folder, 'embeddings.mmap')
        shape = (len(docs), embedding_dim)
        embeddings = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=shape)

        is_cuda = torch.cuda.is_available()
        logger.info(f"Generating embeddings for {len(docs)} documents with batch size {batch_size}")
        for i in tqdm(range(0, len(docs), batch_size), desc="Generating embeddings"):
            end_idx = min(i + batch_size, len(docs))
            batch = docs[i:end_idx]
            if is_cuda:
                with torch.amp.autocast(device_type='cuda'):
                    batch_embed = self.model.encode(
                        batch, 
                        device=self.device,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
            else:
                batch_embed = self.model.encode(
                    batch, device=self.device,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            embeddings[i:end_idx] = batch_embed
            if i % (batch_size * 5) == 0 and is_cuda:
                embeddings.flush()
                torch.cuda.empty_cache()
        logger.info(f"Generated embeddings shape: {embeddings.shape} in {time.time() - start_time:.2f} seconds")
        return embeddings


class TopicModeler:
    """Wraps topic modeling components and handles training and saving of results."""
    def __init__(self, device: str):
        self.device = device
        # Use EmbeddingGenerator to set base batch size
        embedding_gen = EmbeddingGenerator(device)
        self.base_batch_size = embedding_gen.base_batch_size

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
            prediction_data=True,
            core_dist_n_jobs=-1
        )

        self.representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3),
            "POS": PartOfSpeech("en_core_web_sm")
        }
        
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=None,
            min_topic_size=gl.MIN_CLUSTER_SIZE[0],
            calculate_probabilities=False,
            seed_topic_list=gl.SEED_TOPICS,
            verbose=True,
            vectorizer_model=TfidfVectorizer(
                max_df=gl.MAX_DF[0],
                min_df=gl.MIN_DF[0],
                stop_words='english',
                ngram_range=(1, 3),
                use_idf=True,
                smooth_idf=True
            ),
            representation_model={
                "KeyBERT": self.representation_model['KeyBERT'],
                "MMR": self.representation_model['MMR'],
                "POS": self.representation_model['POS']
            },
            nr_topics=gl.NR_TOPICS[0],
            top_n_words=25,  # Number of words per topic
            n_gram_range=(1, 3),  # n-gram range for topic representation
        )
        self.n_topics = None
        
        try:
            api_key_path = os.path.join(os.getcwd(), 'data', 'OPENAI_API_KEY.txt')
            with open(api_key_path, 'r') as f:
                self.client = OpenAI(api_key=f.read().strip())
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise

    def train_topic_model(self, docs: list, embeddings: np.ndarray) -> BERTopic:
        """Train topic model in multiple phases to ensure consistency."""
        start_time = time.time()
        logger.info(f"Starting topic modeling with {len(docs)} documents")
        
        # Ensure all documents are strings
        docs = [str(doc) for doc in docs]
        
        total_docs = len(docs)
        chunk_size = min(200000, total_docs)
        topic_representatives = {}
        global_topic_count = 0

        # Phase 1: Initial topic discovery per chunk
        for i in range(0, total_docs, chunk_size):
            chunk_end = min(i + chunk_size, total_docs)
            logger.info(f"Processing chunk {i}-{chunk_end} of {total_docs}")
            chunk_docs = [str(doc) for doc in docs[i:chunk_end]]  # Ensure chunk docs are strings
            chunk_embeddings = embeddings[i:chunk_end]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            chunk_model = BERTopic(
                umap_model=UMAP(
                    n_neighbors=gl.N_NEIGHBORS[0],
                    n_components=gl.N_COMPONENTS[0],
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42,
                    verbose=False,
                    low_memory=True,
                    n_jobs=-1
                ),
                hdbscan_model=HDBSCAN(
                    min_samples=gl.MIN_SAMPLES[0],
                    min_cluster_size=max(5, gl.MIN_CLUSTER_SIZE[0] // (total_docs // chunk_size)),
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True,
                    core_dist_n_jobs=-1
                ),
                embedding_model=None,
                min_topic_size=gl.MIN_CLUSTER_SIZE[0],
                calculate_probabilities=False,
                nr_topics=gl.NR_TOPICS[0],
                verbose=False
            )

            chunk_topics, _ = chunk_model.fit_transform(chunk_docs, chunk_embeddings)
            topic_docs = chunk_model.get_representative_docs()
            topic_info = chunk_model.get_topic_info()

            for topic_id in topic_info['Topic']:
                if topic_id == -1:
                    continue
                docs_for_topic = topic_docs.get(topic_id, [])[:5]
                doc_indices = [chunk_docs.index(doc) for doc in docs_for_topic if doc in chunk_docs]
                if doc_indices:
                    topic_key = f"topic_{global_topic_count}"
                    global_topic_count += 1
                    topic_representatives[topic_key] = {
                        'docs': [str(chunk_docs[idx]) for idx in doc_indices],  # Ensure representative docs are strings
                        'embeddings': [chunk_embeddings[idx] for idx in doc_indices],
                        'keywords': chunk_model.get_topic(topic_id)
                    }
            del chunk_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Phase 2: Topic Merging and Refinement
        logger.info("Phase 2: Topic merging and refinement")
        all_rep_docs = [str(d) for rep in topic_representatives.values() for d in rep['docs']]  # Ensure all docs are strings
        all_rep_embeddings = np.array([emb for rep in topic_representatives.values() for emb in rep['embeddings']])
        
        if len(all_rep_docs) == 0:
            logger.warning("No representative documents found. Using original documents.")
            all_rep_docs = docs[:min(len(docs), 1000)]  # Use subset of original docs if no representatives
            all_rep_embeddings = embeddings[:min(len(docs), 1000)]

        self.topic_model.fit(all_rep_docs, all_rep_embeddings)

        # Phase 3: Document Mapping
        logger.info("Phase 3: Mapping documents to final topics")
        final_topics = []
        for i in range(0, total_docs, chunk_size):
            chunk_end = min(i + chunk_size, total_docs)
            chunk_docs = [str(doc) for doc in docs[i:chunk_end]]  # Ensure chunk docs are strings
            chunk_embeddings = embeddings[i:chunk_end]
            chunk_topics, _ = self.topic_model.transform(chunk_docs, chunk_embeddings)
            final_topics.extend(chunk_topics)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.final_topics = final_topics
        self.n_topics = len(self.topic_model.get_topics())
        logger.info(f"Topic modeling completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Generated {self.n_topics} unified topics")
        
        # Save the model
        os.makedirs(gl.model_folder, exist_ok=True)
        model_path = os.path.join(gl.model_folder, 
            f"bertopic_model_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}_v8.pkl")
        self.topic_model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return self.topic_model

    def generate_topic_label(self, keywords: list, docs: list = None) -> str:
        """Generate a concise topic label via GPT."""
        try:
            top_keywords = [k.replace('_', ' ') for k in keywords[:5]]
            doc_context = ""
            if docs:
                doc_context = "\nExample discussions:\n" + "\n".join(docs[:2])
            prompt = f"""Generate a concise business topic label (1-2 words) and a subtopic label (2-3 words) based on the provided earnings call keywords.

                ### Examples:
                - **Topic**: "Business Strategy", **Subtopic**: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - **Topic**: "Financial Position", **Subtopic**: "Debt Management", "Liquidity Risk", "Cash Flow", "Working Capital", "Others"
                - **Topic**: "Corporate Governance", **Subtopic**: "Board Structure", "Executive Compensation", "Regulatory Compliance", "Others"   
                - **Topic**: "Technology & Innovation", **Subtopic**: "Artificial Intelligence", "Digital Transformation", "R&D Investment", "Others"
                - **Topic**: "Risk Management", **Subtopic**: "Market Risk", "Operational Risk", "Regulatory Uncertainty", "Financial Stability", "Others"
                - **Topic**: "Market", **Subtopic**: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - **Topic**: "Business Overview", **Subtopic**: "Business Strategy", "Company Description", "Geographic Presence", "Industry Trends", "Market Position", "Product Offerings", "Others"
                - **Topic**: "Contractual Obligations", **Subtopic**: "Revenue", "Earnings Per Share", "Gross Margin", "Net Income", "Others"
                - **Topic**: "Critical Accounting Policies", **Subtopic**: "Allowance for Doubtful Accounts", "Goodwill Impairment", "Income Taxes", "Inventory Valuation", "Revenue Recognition", "Share-Based Compensation", "Others"
                - **Topic**: "Financial Performance", **Subtopic**: "EBITDA", "Earnings Per Share", "Expenses", "Gross Profit", "Net Income", "Operating Income", "Revenues", "Others"
                - **Topic**: "Forward Looking Statements", **Subtopic**: "Assumptions", "Future Outlook", "Growth Strategy", "Market Opportunities", "Potential Risks", "Projections", "Others"
                - **Topic**: "Liquidity and Capital Resources", **Subtopic**: "Capital Expenditures", "Cash Flow", "Credit Facilities", "Debt Management", "Financing Activities", "Investing Activities", "Working Capital", "Others"
                - **Topic**: "Off Balance Sheet Arrangements", **Subtopic**: "Commitments", "Contingent Liabilities", "Guarantees", "Leases", "Variable Interest Entities", "Others"
                - **Topic**: "Recent Accounting Pronouncements", **Subtopic**: "Adoption Impact", "Impact Assessment", "Implementation Plans", "New Standards", "Others"
                - **Topic**: "Recent Developments", **Subtopic**: "Acquisitions", "Divestitures", "New Products", "Strategic Initiatives", "Others"
                - **Topic**: "Regulatory and Legal Matters", **Subtopic**: "Compliance", "Environmental Compliance", "Legal Proceedings", "Legislative Changes", "Regulatory Changes", "Others"
                - **Topic**: "Risk_Factors", **Subtopic**: "Competitive Risks", "Economic Conditions", "Financial Risks", "Market Risks", "Operational Risks", "Regulatory Risks", "Others"
                - **Topic**: "Segment Information", **Subtopic**: "Geographic Segments", "Product Segments", "Customer Segments", "Segment Performance", "Segment Profitability", "Segment Revenue", "Others"
                - **Topic**: "Sustainability_and_CSR", **Subtopic**: "Environmental Impact", "Social Responsibility", "Sustainability Initiatives", "Others"
                - **Topic**: "Accounting Policies", **Subtopic**: "Amortization", "Depreciation", "Revenue Recognition", "Income Taxes", "Leases", "Fair Value", "Goodwill"
                - **Topic**: "Auditor Report", **Subtopic**: "Audit Opinion", "Critical Audit Matters", "Internal Controls", "Basis for Opinion"
                - **Topic**: "Cash Flow", **Subtopic**: "Operating Activities", "Investing Activities", "Financing Activities"
                - **Topic**: "Corporate Governance", **Subtopic**: "Board Structure", "Executive Compensation", "Internal Controls", "Strategic Planning"
                - **Topic**: "Financial Performance", **Subtopic**: "Revenue", "Operating Income", "Net Income", "EPS", "Segment Results"
                - **Topic**: "Financial Position", **Subtopic**: "Assets", "Liabilities", "Equity", "Working Capital", "Investments"
                - **Topic**: "Business Overview", **Subtopic**: "Business Model", "Market Position", "Geographic Presence", "Industry Overview"
                - **Topic**: "Competition", **Subtopic**: "Market Share", "Competitive Advantages", "Industry Trends"
                - **Topic**: "Environmental Risks", **Subtopic**: "Climate Change", "Sustainability", "Resource Management"
                - **Topic**: "External Factors", **Subtopic**: "Economic Conditions", "Geopolitical Risks", "Market Conditions"
                - **Topic**: "Financial Risks", **Subtopic**: "Credit Risk", "Liquidity Risk", "Interest Rate Risk", "Market Risk"
                - **Topic**: "Regulatory Matters", **Subtopic**: "Compliance", "Legal Proceedings", "Regulatory Changes"
                - **Topic**: "Strategic Initiatives", **Subtopic**: "Growth Strategy", "Market Expansion", "Innovation"
                - **Topic**: "Operational Performance", **Subtopic**: "Efficiency", "Productivity", "Cost Management"
                - **Topic**: "Market Analysis", **Subtopic**: "Market Trends", "Consumer Behavior", "Competition"
                - **Topic**: "Industry Specific Information", **Subtopic**: "Industry Policy", "Industry Trends", "Regulatory Environment", "Competitive Landscape", "Others"
                ### Keywords:
                - **Primary Keywords**: {', '.join(top_keywords)}
                - **Secondary Keywords**: {', '.join([k.replace('_', ' ') for k in keywords[5:8]])}

                ### Context:
                {doc_context}

                ### Requirements:
                - Use standard financial and business terminology
                - Ensure topic labels are broad yet meaningful
                - Ensure subtopics are specific and relevant
                - Classify topics based on:
                  1. Financial Fundamentals (performance, position, cash flow)
                  2. Business Operations (strategy, market, competition)
                  3. Governance & Control (policies, audits, compliance)
                  4. Risk Factors (financial, operational, external)
                - Output format: "Topic: [Label], Subtopic: [Specific Area]"
                - Be concise and specific."""

            response = self.client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call analysis and business intelligence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            result = response.choices[0].message.content.strip()
            # Parse the result assuming the output format is "Topic: [Label], Subtopic: [Specific Area]"
            topic_label, subtopic_label = "", ""
            try:
                parts = result.split(",")
                for p in parts:
                    if "Topic" in p:
                            topic_label = p.split(":")[1].strip()
                    elif "Subtopic" in p:
                        subtopic_label = p.split(":")[1].strip()
            except Exception as e:
                logger.error(f"Error parsing topic label: {e}")
                topic_label = ' '.join(top_keywords[:1])
                subtopic_label = ' '.join(top_keywords[1:3])

            return topic_label.strip(), subtopic_label.strip()
        except Exception as e:
            logger.error(f"Error generating topic label: {e}")
            topic_label = ' '.join(top_keywords[:1])
            subtopic_label = ' '.join(top_keywords[1:3])
            return topic_label.strip(), subtopic_label.strip()

    def save_topic_keywords(self, topic_model: BERTopic) -> pd.DataFrame:
        """Generate and save topic keywords with labels."""
        try:
            # Get basic topic info
            topic_info = topic_model.get_topic_info()
            topics_dict = topic_model.get_topics()
            
            # Add different representation methods
            keybert_topics = {topic: topic_model.topic_representations_["KeyBERT"][topic] for topic in topics_dict.keys()}
            mmr_topics = {topic: topic_model.topic_representations_["MMR"][topic] for topic in topics_dict.keys()}
            pos_topics = {topic: topic_model.topic_representations_["POS"][topic] for topic in topics_dict.keys()}
            
            # Add representative documents
            rep_docs = topic_model.representative_docs_
            
            # Create new columns for each representation
            topic_info['KeyBERT'] = topic_info['Topic'].map(lambda x: keybert_topics.get(x, []))
            topic_info['MMR'] = topic_info['Topic'].map(lambda x: mmr_topics.get(x, []))
            topic_info['POS'] = topic_info['Topic'].map(lambda x: pos_topics.get(x, []))
            topic_info['Representative_Docs'] = topic_info['Topic'].map(lambda x: rep_docs.get(x, []))
            
            # Generate labels (split into topic and subtopic)
            main_topics = []
            subtopics = []
            labels = []
            for _, row in tqdm(topic_info.iterrows(), desc="Generating topic labels"):
                if row['Topic'] == -1:
                    labels.append("No Topic")
                    subtopics.append("")
                    continue
                keywords = [word for word, _ in topics_dict.get(row['Topic'], [])]
                docs = row['Representative_Docs']
                topic_label, subtopic_label = self.generate_topic_label(keywords, docs)
                main_topics.append(topic_label)
                subtopics.append(subtopic_label)
            
            topic_info['Topic_Label'] = main_topics
            topic_info['Subtopic_Label'] = subtopics

            output_path = os.path.join(
                gl.output_folder, 
                f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.csv"
            )
            topic_info.to_csv(output_path, index=False)
            logger.info(f"Saved topic keywords with labels to {output_path}")
            return topic_info
        except Exception as e:
            logger.error(f"Error in save_topic_keywords: {e}")
            logger.error(traceback.format_exc())
            raise

    def save_figures(self, topic_model: BERTopic) -> None:
        """Save visualization figures generated by the topic model."""
        try:
            os.makedirs(gl.output_fig_folder, exist_ok=True)
            base_path = os.path.join(gl.output_fig_folder, f'bertopic{gl.num_topic_to_plot}')
            
            # Save barchart
            fig = topic_model.visualize_barchart(top_n_topics=gl.num_topic_to_plot)
            fig.write_image(f"{base_path}.pdf")
            
            # Save intertopic distance map
            fig1 = topic_model.visualize_topics()
            fig1.write_image(f"{base_path}_intertopic_distance_map_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pdf")
            
            # Save heatmap
            fig2 = topic_model.visualize_heatmap()
            fig2.write_image(f"{base_path}_heatmap_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pdf")
            
            # Save hierarchy
            fig3 = topic_model.visualize_hierarchy()
            fig3.write_image(f"{base_path}_hierarchy_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pdf")
            
            logger.info(f"Visualization saved to base path: {base_path}")
        except Exception as e:
            logger.error(f"Error in save_figures: {e}")
            logger.error(traceback.format_exc())


def main() -> None:
    """Main processing pipeline."""
    try:
        device = setup_cuda()
        file_path = os.path.join(os.getcwd(), 'data', gl.data_filename)
        data_handler = DataHandler(file_path, gl.YEAR_START, gl.YEAR_END)

        docs_path = os.path.join(gl.output_folder, f'preprocessed_docs_{gl.YEAR_START}_{gl.YEAR_END}.txt')
        if os.path.exists(docs_path):
            logger.info(f"Found preprocessed docs at {docs_path}. Loading...")
            docs = data_handler.load_doc_parallel(docs_path)
        else:
            logger.info("Processed docs not found. Processing raw data...")
            data = data_handler.load_data()
            docs = data_handler.preprocess_text(data)
            os.makedirs(gl.output_folder, exist_ok=True)
            logger.info(f"Saving processed docs to {docs_path}")
            with open(docs_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(docs))

        embedding_gen = EmbeddingGenerator(device)
        embeddings = embedding_gen.generate_embeddings(docs)

        topic_modeler = TopicModeler(device)
        topic_model = topic_modeler.train_topic_model(docs, embeddings)

        topic_info = topic_modeler.save_topic_keywords(topic_model)
        topic_modeler.save_figures(topic_model)

        logger.info(f"Generated {len(topic_info)} topics")
        logger.info("Topic modeling completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
