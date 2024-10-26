# narrativesBERT
Using Topic Modeling to Study the narratives of earnings calls


# Methodology: Topic Modeling Using BERTopic

In this study, we employ the BERTopic model to extract and analyze thematic structures within a large corpus of textual data. BERTopic, a transformer-based topic modeling technique, leverages contextual embeddings along with clustering algorithms to identify meaningful topics, making it suitable for complex, high-dimensional data. This section outlines the key steps and configurations utilized for implementing BERTopic.

First, we preprocessed the text data to ensure consistency and remove noise. The preprocessing pipeline included tokenization using spaCy, stopword removal, lemmatization, and the generation of n-grams (bigrams and trigrams). These steps helped retain essential terms while removing redundant or irrelevant information. Tokenization and lemmatization were performed using spaCy, with unnecessary components like Named Entity Recognition (NER) and dependency parsing disabled to improve processing efficiency.

After preprocessing, the corpus was used as input for the BERTopic model. To capture semantic relationships, sentence embeddings were generated using the all-MiniLM-L6-v2 model from the SentenceTransformers library. These embeddings were then reduced to a lower-dimensional space using UMAP (Uniform Manifold Approximation and Projection) to facilitate clustering. For this study, UMAP parameters were set to n_neighbors=20, n_components=5, and min_dist=0.2 to balance the local and global structures of the data.

For the clustering step, we utilized HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise), which identifies dense regions within the data to form clusters, corresponding to distinct topics. The min_cluster_size was set to 75 to ensure that the topics identified were significant and not merely small, isolated clusters. This helped minimize the generation of trivial or redundant topics. Additionally, the min_samples parameter was set to 5 to control the density of points needed to define a cluster, ensuring a stable clustering structure.

Finally, the topics were represented using a set of keywords derived from a custom TF-IDF vectorizer. The vectorizer was configured with max_df=0.8, min_df=2, and ngram_range=(1, 1) to emphasize unigrams while preventing the inclusion of overly frequent or rare terms. The output consisted of coherent topic representations, each described by 20 representative keywords, which provided insights into the thematic structures present within the corpus. To further refine the results, we manually merged similar topics to reduce redundancy and increase interpretability.

Overall, BERTopic's integration of transformer-based embeddings, dimensionality reduction, and density-based clustering allowed us to extract coherent and meaningful topics, enabling a detailed exploration of the latent themes within the dataset.

/home/zc_research/
│
├── TAD/
│   └── create_input.py
├── narrativesBERT/
│   ├── __init__.py
│   └── BERTopic_big_data_hpc_v2.py
