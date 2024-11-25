import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import os
from tqdm import tqdm
from pathlib import Path
import pickle
import warnings
import shutil
import gc
warnings.filterwarnings('ignore')

# Memory management settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cuda.max_memory_allocated = 2 * 1024 * 1024 * 1024  # 2GB limit

def clear_memory():
    """Global memory clearing function"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class WatchGraphNetwork:
    def __init__(self, config=None):
        self.config = {
            'price_similarity_threshold': 0.1,  
            'size_similarity_threshold': 2,     
            'embedding_dim': 128,               
            'temporal_embedding_dim': 32,       
            'material_embedding_dim': 64,       
            'brand_embedding_dim': 128,         
            'movement_embedding_dim': 64,       
            'similarity_metric': 'cosine',      
            'edge_weight_alpha': 0.5,          
            'min_connections': 3,              
            'max_edges_per_node': 5,          
            'chunk_size': 50,                 
            'similarity_threshold': 0.7,      
            'window_size': 1000,             
            'condition_weights': {
                'New': 1.0,
                'Unworn': 0.95,
                'Very good': 0.8,
                'Good': 0.7,
                'Fair': 0.5
            },
            'num_gnn_layers': 3,               
            'hidden_channels': 256,             
            'dropout_rate': 0.2,               
            'aggregation_type': 'mean',        
            'learning_rate': 0.001,            
            'weight_decay': 5e-4,              
            'batch_size': 32,                  
            'num_epochs': 100,                 
            'edge_embedding_dim': 16,          
            'use_edge_features': True,         
            'use_attention': True,             
            'num_heads': 4,                    
            'residual_connections': True,      
            'layer_norm': True                 
        } if config is None else config

        self.price_scaler = MinMaxScaler()
        self.size_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create necessary directories
        for directory in ['checkpoints', 'checkpoints/temp']:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def clear_memory():
        clear_memory()
    def load_data(self, filepath=None):
        """Load watch dataset with checkpointing"""
        if filepath is None:
            filepath = Path("/Watches.csv")
        
        checkpoint_path = 'checkpoints/loaded_data.pkl'
        if os.path.exists(checkpoint_path):
            print("Loading data from checkpoint...")
            return pd.read_pickle(checkpoint_path)
        
        try:
            print(f"Loading data from: {filepath}")
            df = pd.read_csv(
                filepath,
                encoding='utf-8',
                low_memory=False,
                on_bad_lines='skip'
            )
            
            print(f"Successfully loaded {len(df)} rows")
            df.to_pickle(checkpoint_path)
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Checking for CSV files in current directory:")
            for file in os.listdir():
                if file.endswith('.csv'):
                    print(f"Found: {file}")
            return None

    def clean_price(self, price):
        """Clean price values with better error handling"""
        try:
            if pd.isna(price) or price == 'Price on request':
                return np.nan
            price_str = str(price).replace('$', '').replace(',', '').strip()
            return float(price_str) if price_str else np.nan
        except:
            return np.nan

    def clean_size(self, size):
        """Clean size values with better error handling"""
        try:
            if pd.isna(size):
                return np.nan
            size_str = str(size).lower().replace('mm', '').strip()
            return float(size_str) if size_str else np.nan
        except:
            return np.nan

    def create_condition_score(self, condition):
        """Create numerical score for condition"""
        if pd.isna(condition):
            return 0.5
        condition_str = str(condition).strip()
        return self.config['condition_weights'].get(condition_str, 0.5)

    def create_temporal_embedding(self, year):
        """Create temporal embedding with better error handling"""
        try:
            if pd.isna(year):
                return np.zeros(2)
            year_str = str(year).split()[0]
            year_val = float(year_str) if year_str.replace('.', '').isdigit() else np.nan
            if np.isnan(year_val):
                return np.zeros(2)
            return np.array([
                np.sin(2 * np.pi * year_val / 100),
                np.cos(2 * np.pi * year_val / 100)
            ])
        except:
            return np.zeros(2)

    def create_material_embedding(self, material):
        """Create material embedding with memory efficiency"""
        try:
            if pd.isna(material) or material == '':
                return np.zeros(self.config['material_embedding_dim'])
            embedding = self.sentence_model.encode(
                str(material),
                show_progress_bar=False,
                normalize_embeddings=True
            )[:self.config['material_embedding_dim']]
            return embedding
        except:
            return np.zeros(self.config['material_embedding_dim'])

    def create_brand_embedding(self, brand, avg_price):
        """Create brand embedding with memory efficiency"""
        try:
            if pd.isna(brand):
                return np.zeros(self.config['brand_embedding_dim'] + 1)
            brand_emb = self.sentence_model.encode(
                str(brand),
                show_progress_bar=False,
                normalize_embeddings=True
            )[:self.config['brand_embedding_dim']]
            price_factor = np.array([avg_price if not pd.isna(avg_price) else 0])
            return np.concatenate([brand_emb, price_factor])
        except:
            return np.zeros(self.config['brand_embedding_dim'] + 1)

    def create_movement_embedding(self, movement):
        """Create movement embedding with memory efficiency"""
        try:
            if pd.isna(movement) or movement == '':
                return np.zeros(self.config['movement_embedding_dim'])
            embedding = self.sentence_model.encode(
                str(movement),
                show_progress_bar=False,
                normalize_embeddings=True
            )[:self.config['movement_embedding_dim']]
            return embedding
        except:
            return np.zeros(self.config['movement_embedding_dim'])

    def compute_similarity(self, node1, node2):
        """Compute cosine similarity between two nodes"""
        try:
            norm1 = np.linalg.norm(node1)
            norm2 = np.linalg.norm(node2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return np.dot(node1, node2) / (norm1 * norm2)
        except:
            return 0
        
    def create_watch_to_watch_edges_simple(self, node_features):
        """Create edges using simple similarity computation with extreme memory efficiency"""
        print("Starting edge creation...")
        edges_checkpoint = 'checkpoints/edges.npz'
        
        if os.path.exists(edges_checkpoint):
            print("Loading edges from checkpoint...")
            edges_data = np.load(edges_checkpoint)
            return edges_data['edges'], edges_data['weights']
        
        print("Converting features...")
        features = np.array(node_features, dtype=np.float32)
        n_nodes = len(features)
        chunk_size = self.config['chunk_size']
        k = self.config['min_connections']
        window_size = self.config['window_size']
        
        edges = []
        edge_weights = []
        
        try:
            for i in tqdm(range(0, n_nodes, chunk_size), desc="Processing nodes"):
                chunk_start = i
                chunk_end = min(i + chunk_size, n_nodes)
                chunk = features[chunk_start:chunk_end]
                
                # Process each node in chunk
                for idx, node in enumerate(chunk):
                    global_idx = chunk_start + idx
                    
                    # Define window for comparison
                    window_start = max(0, global_idx - window_size//2)
                    window_end = min(n_nodes, global_idx + window_size//2)
                    
                    # Get comparison nodes
                    comp_nodes = features[window_start:window_end]
                    
                    # Compute similarities
                    similarities = []
                    for j, other_node in enumerate(comp_nodes):
                        other_idx = window_start + j
                        if other_idx != global_idx:
                            sim = self.compute_similarity(node, other_node)
                            if sim > self.config['similarity_threshold']:
                                similarities.append((other_idx, sim))
                    
                    # Sort and keep top k similarities
                    if similarities:
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        for target_idx, sim in similarities[:k]:
                            edges.append([global_idx, target_idx])
                            edge_weights.append(float(sim))
                    
                    # Clear memory periodically
                    if idx % 10 == 0:
                        self.clear_memory()
                
                # Save checkpoint periodically
                if len(edges) > 0 and i % (chunk_size * 10) == 0:
                    temp_edges = np.array(edges)
                    temp_weights = np.array(edge_weights)
                    np.savez(edges_checkpoint, edges=temp_edges, weights=temp_weights)
                
                # Clear memory after each chunk
                self.clear_memory()
        
        except Exception as e:
            print(f"Error during edge creation: {str(e)}")
            if len(edges) > 0:
                edges = np.array(edges)
                edge_weights = np.array(edge_weights)
                np.savez(edges_checkpoint, edges=edges, weights=edge_weights)
                return edges, edge_weights
            raise
        
        # Convert to final format
        print("Converting to final format...")
        edges = np.array(edges)
        edge_weights = np.array(edge_weights)
        
        # Save final results
        print("Saving final results...")
        np.savez(edges_checkpoint, edges=edges, weights=edge_weights)
        print(f"Created {len(edges)} edges")
        
        return edges, edge_weights

    def preprocess_data(self, df):
        """Preprocess data with enhanced memory efficiency and checkpointing"""
        feature_checkpoint = 'checkpoints/features.npy'
        processed_df_checkpoint = 'checkpoints/processed_df.pkl'
        
        try:
            if os.path.exists(feature_checkpoint) and os.path.exists(processed_df_checkpoint):
                print("Loading preprocessed data from checkpoints...")
                watch_features = np.load(feature_checkpoint)
                df = pd.read_pickle(processed_df_checkpoint)
                print("Successfully loaded preprocessed data")
            else:
                print("Starting preprocessing from scratch...")
                df = df.copy()
                
                print("Processing numerical features...")
                df['price_clean'] = df['price'].apply(self.clean_price)
                df['size_clean'] = df['size'].apply(self.clean_size)
                
                print("Handling missing values...")
                price_mean = df['price_clean'].mean()
                size_mean = df['size_clean'].mean()
                df['price_clean'] = df['price_clean'].fillna(price_mean)
                df['size_clean'] = df['size_clean'].fillna(size_mean)
                
                print("Scaling features...")
                df['price_scaled'] = self.price_scaler.fit_transform(df[['price_clean']])
                df['size_scaled'] = self.size_scaler.fit_transform(df[['size_clean']])
                
                # Process embeddings in chunks
                print("Processing embeddings...")
                chunk_size = 1000
                total_rows = len(df)
                
                for start_idx in tqdm(range(0, total_rows, chunk_size), desc="Processing chunks"):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    chunk_df = df.iloc[start_idx:end_idx].copy()
                    
                    # Create embeddings for chunk
                    chunk_df['condition_score'] = chunk_df['condition'].apply(self.create_condition_score)
                    chunk_df['temporal_embedding'] = chunk_df['yop'].apply(self.create_temporal_embedding)
                    chunk_df['material_embedding'] = chunk_df['casem'].apply(self.create_material_embedding)
                    chunk_df['movement_embedding'] = chunk_df['mvmt'].apply(self.create_movement_embedding)
                    
                    # Update main dataframe
                    for col in ['condition_score', 'temporal_embedding', 'material_embedding', 'movement_embedding']:
                        df.loc[start_idx:end_idx, col] = chunk_df[col]
                    
                    self.clear_memory()
                
                # Process brand embeddings
                print("Creating brand embeddings...")
                brand_avg_prices = df.groupby('brand')['price_clean'].mean()
                
                # Process brand embeddings in chunks
                for start_idx in tqdm(range(0, total_rows, chunk_size), desc="Processing brand embeddings"):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    chunk_df = df.iloc[start_idx:end_idx].copy()
                    
                    chunk_df['brand_embedding'] = chunk_df.apply(
                        lambda x: self.create_brand_embedding(
                            x['brand'],
                            brand_avg_prices.get(x['brand'], 0)
                        ),
                        axis=1
                    )
                    
                    df.loc[start_idx:end_idx, 'brand_embedding'] = chunk_df['brand_embedding']
                    self.clear_memory()

                    # Create feature matrix
                print("Creating feature matrix...")
                watch_features = np.column_stack([
                    df['price_scaled'],
                    df['size_scaled'],
                    df['condition_score'],
                    np.vstack(df['temporal_embedding'].values),
                    np.vstack(df['material_embedding'].values),
                    np.vstack(df['movement_embedding'].values),
                    np.vstack(df['brand_embedding'].values)
                ])
                
                # Save checkpoints
                print("Saving checkpoints...")
                np.save(feature_checkpoint, watch_features)
                df.to_pickle(processed_df_checkpoint)
            
            # Create edges
            print("Creating edges...")
            edges, edge_weights = self.create_watch_to_watch_edges_simple(watch_features)
            
            # Create PyG Data object
            print("Creating PyG Data object...")
            data = Data(
                x=torch.FloatTensor(watch_features),
                edge_index=torch.LongTensor(edges.T),
                edge_attr=torch.FloatTensor(edge_weights.reshape(-1, 1))
            )
            
            # Save final graph data
            print("Saving graph data...")
            torch.save(data, 'checkpoints/watch_gnn_data.pt')
            
            return data, df
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None, None

    def create_gnn_model(self, in_channels):
        """Create GNN model with proper dimension handling"""
        class WatchGNN(nn.Module):
            def __init__(self, in_channels, config):
                super(WatchGNN, self).__init__()
                self.config = config
                
                # Calculate dimensions for each layer
                self.hidden_dim = self.config['hidden_channels']
                self.num_heads = self.config['num_heads']
                
                if self.config['use_attention']:
                    # First layer: in_channels -> hidden_dim * num_heads
                    self.conv1 = GATConv(
                        in_channels, 
                        self.hidden_dim,
                        heads=self.num_heads,
                        dropout=self.config['dropout_rate']
                    )
                    
                    # Second layer: (hidden_dim * num_heads) -> hidden_dim * num_heads
                    self.conv2 = GATConv(
                        self.hidden_dim * self.num_heads,
                        self.hidden_dim,
                        heads=self.num_heads,
                        dropout=self.config['dropout_rate']
                    )
                    
                    # Final layer: (hidden_dim * num_heads) -> hidden_dim
                    self.conv3 = GATConv(
                        self.hidden_dim * self.num_heads,
                        self.hidden_dim,
                        heads=1,
                        dropout=self.config['dropout_rate']
                    )
                else:
                    self.conv1 = GCNConv(in_channels, self.hidden_dim)
                    self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
                    self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
                
                self.dropout = nn.Dropout(self.config['dropout_rate'])
                
                # Layer normalization with correct dimensions
                if self.config['layer_norm']:
                    self.norm1 = nn.LayerNorm(self.hidden_dim * self.num_heads)
                    self.norm2 = nn.LayerNorm(self.hidden_dim * self.num_heads)
                    self.norm3 = nn.LayerNorm(self.hidden_dim)  # Final output dimension

            def forward(self, x, edge_index, edge_attr=None):
                # First layer
                x1 = self.conv1(x, edge_index)
                if self.config['layer_norm']:
                    x1 = self.norm1(x1)
                x1 = F.relu(x1)
                x1 = self.dropout(x1)
                
                # Second layer
                x2 = self.conv2(x1, edge_index)
                if self.config['layer_norm']:
                    x2 = self.norm2(x2)
                if self.config['residual_connections']:
                    # Ensure dimensions match before residual connection
                    if x1.size(-1) == x2.size(-1):
                        x2 = x2 + x1
                x2 = F.relu(x2)
                x2 = self.dropout(x2)
                
                # Third layer
                x3 = self.conv3(x2, edge_index)
                if self.config['layer_norm']:
                    x3 = self.norm3(x3)
                if self.config['residual_connections']:
                    # Only add residual if dimensions match
                    if x2.size(-1) == x3.size(-1):
                        x3 = x3 + x2
                
                return F.relu(x3)

        return WatchGNN(in_channels, self.config)

    def process_and_train(self):
        """Main processing and training pipeline with enhanced checkpointing"""
        print("Starting data processing pipeline...")
        
        try:
            # Check for final results
            if (os.path.exists('checkpoints/final_embeddings.pt') and
                os.path.exists('checkpoints/watch_gnn_data.pt') and
                os.path.exists('checkpoints/processed_df.pkl')):
                print("Loading final results from checkpoints...")
                node_embeddings = torch.load('checkpoints/final_embeddings.pt')
                graph_data = torch.load('checkpoints/watch_gnn_data.pt')
                processed_df = pd.read_pickle('checkpoints/processed_df.pkl')
                return node_embeddings, graph_data, processed_df
            
            print("Loading and preprocessing data...")
            df = self.load_data()
            if df is None:
                return None, None, None
            
            print("Preprocessing data...")
            graph_data, processed_df = self.preprocess_data(df)
            if graph_data is None:
                return None, None, None
            
            print("Creating and initializing GNN model...")
            try:
                model = self.create_gnn_model(in_channels=graph_data.x.size(1))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {device}")
                
                # Move to CPU for memory efficiency
                model = model.to('cpu')
                graph_data = graph_data.to('cpu')
                
                print("Generating embeddings...")
                model.eval()
                with torch.no_grad():
                    node_embeddings = model(
                        graph_data.x,
                        graph_data.edge_index,
                        graph_data.edge_attr
                    )
                
                print("Saving embeddings...")
                torch.save(node_embeddings, 'checkpoints/final_embeddings.pt')
                
                return node_embeddings, graph_data, processed_df
                
            except Exception as e:
                print(f"Error in model creation or embedding generation: {str(e)}")
                return None, None, None
                
        except Exception as e:
            print(f"Error in processing pipeline: {str(e)}")
            return None, None, None

def main():
    try:
        print("Initializing WatchGraphNetwork...")
        watch_gnn = WatchGraphNetwork()
        
        print("Starting processing and training...")
        node_embeddings, graph_data, processed_df = watch_gnn.process_and_train()
        
        if all(v is not None for v in [node_embeddings, graph_data, processed_df]):
            print("\nSuccess! Network creation complete.")
            print(f"Node embeddings shape: {node_embeddings.shape}")
            print(f"Number of edges: {graph_data.edge_index.shape[1]}")
            print(f"Number of watches: {len(processed_df)}")
            print("\nSaved files in checkpoints/:")
            print("- processed_df.pkl")
            print("- watch_gnn_data.pt")
            print("- final_embeddings.pt")
        else:
            print("\nFailed to create network. Check the error messages above.")
    
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()