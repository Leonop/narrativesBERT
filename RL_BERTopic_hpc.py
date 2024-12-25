import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
import global_options as gl
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from bertopic import BERTopic


class RL_BERTopicGPU:
    """
    Reinforcement Learning Enhanced BERTopic Model
    
    Logic Flow:
    1. Initialization Phase:
        - Initialize BERTopic components
        - Setup RL environment
        - Initialize tracking metrics
    
    2. Training Phase:
        - Document embedding
        - Parameter optimization via RL
        - Topic modeling with optimized parameters
    
    3. Evaluation Phase:
        - Topic coherence assessment
        - Diversity measurement
        - Coverage analysis
    
    Complexity Analysis:
    - Time Complexity:
        * Document Embedding: O(N * D), where N = number of documents, D = embedding dimension
        * RL Training: O(E * S * A), where E = episodes, S = steps per episode, A = action space size
        * Topic Modeling: O(N * log N) for clustering
        Total: O(N * D + E * S * A + N * log N)
    
    - Space Complexity:
        * Document Embeddings: O(N * D)
        * RL Memory Buffer: O(M), where M = memory buffer size
        * Topic Model: O(K * V), where K = number of topics, V = vocabulary size
        Total: O(N * D + M + K * V)
    """
    
    def __init__(self, 
                 state_size=3,           # [coherence, diversity, coverage]
                 action_size=8,          # parameter adjustment actions
                 memory_size=10000,      # RL memory buffer size
                 batch_size=32,          # RL training batch size
                 gamma=0.95,             # discount factor
                 learning_rate=0.001):
        """Initialize RL-enhanced BERTopic model"""
        # Initialize base BERTopic components
        self.init_bertopic_components()
        
        # Initialize RL components
        self.init_rl_components(state_size, action_size, memory_size, 
                              batch_size, gamma, learning_rate)
        
        # Initialize tracking metrics
        self.init_tracking_metrics()
        
        # Setup visualization directory
        self.setup_viz_directory()

    def init_bertopic_components(self):
        """Initialize BERTopic model components"""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2', 
            device='cuda'
        )
        self.embedding_model.max_seq_length = 512
        
        # Initialize UMAP
        self.umap_model = UMAP(
            n_components=gl.N_COMPONENTS[0],
            n_neighbors=gl.N_NEIGHBORS[0],
            random_state=42,
            metric=gl.METRIC[0],
            verbose=True,
            low_memory=False,
            n_jobs=-1
        )
        
        # Initialize HDBSCAN
        self.hdbscan_model = HDBSCAN(
            min_samples=gl.MIN_SAMPLES[0],
            min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
            prediction_data=True,
            core_dist_n_jobs=-1
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init_rl_components(self, state_size, action_size, memory_size, 
                          batch_size, gamma, learning_rate):
        """Initialize RL components"""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize policy network
        self.policy_network = self.build_policy_network(state_size, action_size)
        self.target_network = self.build_policy_network(state_size, action_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), 
                                  lr=learning_rate)

    def init_tracking_metrics(self):
        """Initialize metrics for tracking model performance"""
        self.training_history = {
            'coherence_scores': [],
            'diversity_scores': [],
            'coverage_scores': [],
            'rewards': [],
            'parameters': [],
            'epsilon_values': [],
            'loss_values': [],
            'topic_quality': []
        }

    def train_rl_bertopic(self, docs, n_episodes=5, steps_per_episode=10):
        """
        Train the RL-enhanced BERTopic model
        
        Args:
            docs: List of documents
            n_episodes: Number of RL episodes
            steps_per_episode: Steps per episode
            
        Returns:
            best_model: Trained BERTopic model with best parameters
            training_history: Training metrics history
        """
        best_reward = float('-inf')
        best_params = None
        best_model = None
        
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            
            # Train for one episode
            episode_reward, params, model = self.train_episode(
                docs, 
                steps_per_episode
            )
            
            # Update best model if needed
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = params
                best_model = model
            
            # Visualize progress
            self.visualize_training_progress(episode)
            
            # Save intermediate results
            self.save_training_state(episode)
            
        return best_model, self.training_history

    def visualize_training_progress(self, episode):
        """Create visualizations of training progress"""
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 3)
        
        # 1. Learning Curves
        self.plot_learning_curves(fig.add_subplot(gs[0, 0]))
        
        # 2. Reward History
        self.plot_reward_history(fig.add_subplot(gs[0, 1]))
        
        # 3. Parameter Evolution
        self.plot_parameter_evolution(fig.add_subplot(gs[0, 2]))
        
        # 4. Topic Quality Metrics
        self.plot_topic_quality(fig.add_subplot(gs[1, 0]))
        
        # 5. Epsilon Decay
        self.plot_epsilon_decay(fig.add_subplot(gs[1, 1]))
        
        # 6. Loss Values
        self.plot_loss_values(fig.add_subplot(gs[1, 2]))
        
        # 7. Topic Coherence Distribution
        self.plot_topic_coherence_dist(fig.add_subplot(gs[2, 0]))
        
        # 8. Topic Diversity Matrix
        self.plot_topic_diversity_matrix(fig.add_subplot(gs[2, 1]))
        
        # 9. Model Performance Summary
        self.plot_performance_summary(fig.add_subplot(gs[2, 2]))
        
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.viz_dir, 
            f'training_progress_episode_{episode}.pdf'
        ))
        plt.close()

    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Implementation of interactive dashboard
        pass

    def save_training_state(self, episode):
        """Save training state and metrics"""
        # Save training history
        with open(os.path.join(
            gl.output_folder, 
            f'training_history_episode_{episode}.json'
        ), 'w') as f:
            json.dump(self.training_history, f)
        
        # Save model state
        torch.save(self.policy_network.state_dict(), 
                  os.path.join(gl.model_folder, 
                             f'rl_policy_episode_{episode}.pt'))
        

    def build_policy_network(self, state_size, action_size):
        """Build the policy network for RL"""
        return nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(self.device)

    def setup_viz_directory(self):
        """Setup visualization directory"""
        self.viz_dir = os.path.join(gl.output_fig_folder, 'rl_training')
        os.makedirs(self.viz_dir, exist_ok=True)

    def train_episode(self, docs, steps_per_episode):
        """Train for one episode"""
        current_params = {
            'n_neighbors': gl.N_NEIGHBORS[0],
            'n_components': gl.N_COMPONENTS[0],
            'min_cluster_size': gl.MIN_CLUSTER_SIZE[0],
            'min_samples': gl.MIN_SAMPLES[0]
        }
        
        episode_rewards = []
        
        for step in range(steps_per_episode):
            # Get current state
            state = self.get_state(docs, current_params)
            
            # Select action
            action = self.select_action(state)
            
            # Apply action and get new parameters
            new_params = self.apply_action(action, current_params)
            
            # Get new state and reward
            new_state = self.get_state(docs, new_params)
            reward = self.calculate_reward(state, new_state)
            
            # Store transition
            self.memory.append((state, action, reward, new_state))
            
            # Update policy
            loss = self.update_policy()
            
            # Update tracking metrics
            self.update_training_metrics(state, reward, current_params, loss)
            
            # Update current parameters
            current_params = new_params
            episode_rewards.append(reward)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, 
                            self.epsilon * self.epsilon_decay)
        
        # Train final model with current parameters
        model = self.train_bertopic(docs, current_params)
        
        return np.mean(episode_rewards), current_params, model

    def plot_learning_curves(self, ax):
        """Plot learning curves"""
        ax.plot(self.training_history['coherence_scores'], label='Coherence')
        ax.plot(self.training_history['diversity_scores'], label='Diversity')
        ax.plot(self.training_history['coverage_scores'], label='Coverage')
        ax.set_title('Learning Curves')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        ax.legend()

    def plot_reward_history(self, ax):
        """Plot reward history"""
        ax.plot(self.training_history['rewards'])
        ax.set_title('Reward History')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')

    def get_state(self, docs, params):
        """Get current state representation"""
        model = self.train_bertopic(docs, params)
        coherence = self.compute_coherence(model, docs)
        diversity = self.compute_diversity(model)
        coverage = self.compute_coverage(model, docs)
        return np.array([coherence, diversity, coverage])

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_network(state_tensor).argmax().item()

    def calculate_reward(self, state, new_state):
        """Calculate reward based on state improvement"""
        coherence_improvement = new_state[0] - state[0]
        diversity_improvement = new_state[1] - state[1]
        coverage_improvement = new_state[2] - state[2]
        
        # Add weights and normalize
        weighted_improvement = (
            coherence_improvement * 0.4 + 
            diversity_improvement * 0.4 + 
            coverage_improvement * 0.2
        )
        
        # Add penalty for extreme parameter values
        return weighted_improvement  # Remove the ... at the end
        
    def train_rl_bertopic(self, docs, n_episodes=5, steps_per_episode=10):
        """Train the RL-enhanced BERTopic model"""
        try:
            best_reward = float('-inf')
            best_params = None
            best_model = None
            
            for episode in range(n_episodes):
                print(f"\nEpisode {episode + 1}/{n_episodes}")
                
                try:
                    episode_reward, params, model = self.train_episode(
                        docs, 
                        steps_per_episode
                    )
                    
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        best_params = params
                        best_model = model
                    
                    self.visualize_training_progress(episode)
                    self.save_training_state(episode)
                    
                except Exception as e:
                    print(f"Error in episode {episode}: {str(e)}")
                    continue
            
            return best_model, self.training_history
        
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return None, self.training_history
        
    def compute_coherence(self, model, docs):
        """Compute topic coherence score"""
        topics = model.get_topics()
        # Convert topics to format required for coherence calculation
        topic_words = [[word for word, _ in topic] for topic in topics.values()]
        
        # Calculate coherence using gensim
        texts = [doc.split() for doc in docs]  # Tokenize docs
        dictionary = Dictionary(texts)
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()

    def compute_diversity(self, model):
        """Compute topic diversity score"""
        topics = model.get_topics()
        unique_words = set()
        total_words = 0
        
        for topic in topics.values():
            words = [word for word, _ in topic[:10]]  # Top 10 words
            unique_words.update(words)
            total_words += len(words)
        
        return len(unique_words) / total_words if total_words > 0 else 0

    def compute_coverage(self, model, docs):
        """Compute topic coverage score"""
        topics, _ = model.transform(docs)
        assigned_docs = sum(1 for topic in topics if topic != -1)
        return assigned_docs / len(docs)

    def update_training_metrics(self, state, reward, params, loss):
        """Update training history with current metrics"""
        self.training_history['coherence_scores'].append(state[0])
        self.training_history['diversity_scores'].append(state[1])
        self.training_history['coverage_scores'].append(state[2])
        self.training_history['rewards'].append(reward)
        self.training_history['parameters'].append(list(params.values()))
        self.training_history['epsilon_values'].append(self.epsilon)
        self.training_history['loss_values'].append(loss.item() if loss is not None else 0)


    def plot_parameter_evolution(self, ax):
        """Plot parameter evolution"""
        params = np.array(self.training_history['parameters'])
        param_names = ['n_neighbors', 'n_components', 
                    'min_cluster_size', 'min_samples']
        
        for i, name in enumerate(param_names):
            ax.plot(params[:, i], label=name)
        
        ax.set_title('Parameter Evolution')
        ax.set_xlabel('Step')
        ax.set_ylabel('Parameter Value')
        ax.legend()

    def plot_topic_quality(self, ax):
        """Plot topic quality metrics"""
        quality_metrics = {
            'Avg Coherence': np.mean(self.training_history['coherence_scores']),
            'Max Coherence': np.max(self.training_history['coherence_scores']),
            'Avg Diversity': np.mean(self.training_history['diversity_scores']),
            'Topic Coverage': self.training_history['coverage_scores'][-1]
        }
        
        ax.bar(range(len(quality_metrics)), list(quality_metrics.values()))
        ax.set_xticks(range(len(quality_metrics)))
        ax.set_xticklabels(quality_metrics.keys(), rotation=45)
        ax.set_title('Topic Quality Metrics')


    def train_bertopic(self, docs, params):
        """Train BERTopic model with given parameters"""
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Update model parameters
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
                core_dist_n_jobs=-1
            )
            
            # Create and train model
            model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model
            )
            
            model.fit_transform(docs)
            return model
            
        except Exception as e:
            print(f"Error in training BERTopic: {str(e)}")
            return None

    