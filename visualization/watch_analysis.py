import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import umap
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain
import os
from torch_geometric.data import Data
from pathlib import Path
import squarify 
from scipy.stats import gaussian_kde
from datetime import datetime
import calendar
import plotly.colors as colors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Allow PyTorch Geometric Data loading
torch.serialization.add_safe_globals([Data])

class WatchEmbeddingAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer.
        Sets up paths, loads data, and initializes visualization parameters.
        """
        print("Loading data...")
        
        # Setup paths
        self._setup_paths()
        
        # Setup visualization parameters
        self.viz_params = {
            'width': 1200,
            'height': 800,
            'template': 'plotly_white',
            'color_scheme': px.colors.qualitative.Set3,
            'sample_size': 50000
        }
        
        # Load data
        self._load_data()
        
    def _setup_paths(self):
        """
        Setup all necessary paths for data loading and saving.
        Creates required directories if they don't exist.
        """
        # Set base paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.dirname(current_dir)
        
        # Set specific paths
        self.checkpoints_dir = os.path.join(self.base_path, "checkpoints") # if you have checkpoints from running the code again.
        self.embedding_path = os.path.join(self.checkpoints_dir, "final_embeddings.pt")
        self.graph_data_path = os.path.join(self.checkpoints_dir, "watch_gnn_data.pt")
        self.processed_df_path = os.path.join(self.checkpoints_dir, "processed_df.pkl")
        
        # Setup visualization directories
        self.output_dir = os.path.join(self.base_path, "analysis_results")
        self.viz_dirs = {
            'eda': os.path.join(self.output_dir, 'eda'),
            'network': os.path.join(self.output_dir, 'network'),
            'brand': os.path.join(self.output_dir, 'brand'),
            'price': os.path.join(self.output_dir, 'price'),
            'temporal': os.path.join(self.output_dir, 'temporal'),
            'embedding': os.path.join(self.output_dir, 'embedding'),
        }
        
        # Create directories
        for dir_path in self.viz_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    def _load_data(self):
        """
        Load and validate all required data files.
        Includes error handling and data validation.
        """
        try:
            print("\nLoading embeddings...")
            self.embeddings = torch.load(self.embedding_path).detach().cpu().numpy()
            print("Loaded embeddings successfully")
            
            print("Loading graph data...")
            self.graph_data = torch.load(self.graph_data_path)
            print("Loaded graph data successfully")
            
            print("Loading processed DataFrame...")
            self.df = pd.read_pickle(self.processed_df_path)
            
            # Enhance DataFrame with additional features
            self._enhance_dataframe()
            
            print(f"Loaded {len(self.df)} watches with {self.embeddings.shape[1]} dimensional embeddings")
            
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            raise

    def _enhance_dataframe(self):
        """
        Enhance the DataFrame with additional features for visualization.
        Creates derived features and normalizes data for better visualization.
        """
        # Add price segments
        self.df['price_segment'] = pd.qcut(self.df['price_clean'], 
                                         q=5, 
                                         labels=['Entry', 'Mid-Low', 'Mid', 'Mid-High', 'Luxury'])
        
        # Add year segments
        self.df['year'] = pd.to_numeric(self.df['yop'].str.split().str[0], errors='coerce')
        self.df['decade'] = (self.df['year'] // 10) * 10
        
        # Clean and normalize size data
        self.df['size_segment'] = pd.qcut(self.df['size_clean'], 
                                        q=4, 
                                        labels=['Small', 'Medium', 'Large', 'Extra Large'])

    def generate_eda_visualizations(self):
        """
        Generate  exploratory data analysis visualizations.
        Creates a suite of plots for understanding the dataset.
        """
        print("Generating EDA visualizations...")
        
        try:
            # 1. Price Distribution Analysis
            self._visualize_price_distribution()
            print("Completed price distribution visualization")
            
            # 2. Brand Analysis
            self._visualize_brand_distribution()
            print("Completed brand distribution visualization")
            
            # 3. Feature Correlations
            self._visualize_feature_correlations()
            print("Completed feature correlations visualization")
            
            # 4. Size Analysis
            self._visualize_size_distribution()
            print("Completed size distribution visualization")
            
            print("EDA visualizations completed successfully")
            
        except Exception as e:
            print(f"Error in generate_eda_visualizations: {str(e)}")
            raise

    def _visualize_size_distribution(self):
        """
        Create visualization of watch size distribution.
        """
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=['Size Distribution',
                                        'Size by Brand (Top 10)',
                                        'Size Segments',
                                        'Size vs Price'])
        
        # Size histogram
        fig.add_trace(
            go.Histogram(x=self.df['size_clean'],
                        name='Size Distribution',
                        nbinsx=50),
            row=1, col=1
        )
        
        # Size by top brands
        top_brands = self.df['brand'].value_counts().head(10).index
        brand_data = self.df[self.df['brand'].isin(top_brands)]
        fig.add_trace(
            go.Box(x=brand_data['brand'],
                y=brand_data['size_clean'],
                name='Size by Brand'),
            row=1, col=2
        )
        
        # Size segments
        fig.add_trace(
            go.Bar(x=self.df['size_segment'].value_counts().index,
                y=self.df['size_segment'].value_counts().values,
                name='Size Segments'),
            row=2, col=1
        )
        
        # Size vs Price scatter
        fig.add_trace(
            go.Scatter(x=self.df['size_clean'],
                    y=self.df['price_clean'],
                    mode='markers',
                    opacity=0.5,
                    name='Size vs Price'),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, width=1200,
                        title_text="Size Analysis Dashboard")
        fig.write_html(os.path.join(self.viz_dirs['eda'], 'size_analysis.html'))


    def verify_file_structure(self):
        """
        Verify all required files and directories exist.
        Returns True if all files are found, False otherwise.
        """
        required_files = {
            'Embeddings': self.embedding_path,
            'Graph Data': self.graph_data_path,
            'Processed DataFrame': self.processed_df_path
        }
        
        all_exist = True
        print("\nVerifying file structure:")
        for name, path in required_files.items():
            exists = os.path.exists(path)
            print(f"{name}: {'Found' if exists else 'Not Found'} at {path}")
            if not exists:
                all_exist = False
        
        # Verify output directories
        for dir_name, dir_path in self.viz_dirs.items():
            exists = os.path.exists(dir_path)
            print(f"Output directory '{dir_name}': {'Found' if exists else 'Created'}")
            if not exists:
                os.makedirs(dir_path, exist_ok=True)
        
        return all_exist
    
    def _visualize_price_distribution(self):
        """
        Create price distribution visualizations.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Price Distribution (Log Scale)',
                'Price by Brand (Top 10)',
                'Price Segments Distribution',
                'Price Trend by Year'
            ]
        )
        
        # 1. Price Distribution with log scale
        log_prices = np.log10(self.df['price_clean'] + 1)  # Add 1 to handle zeros
        
        fig.add_trace(
            go.Histogram(
                x=log_prices,
                name='Price Distribution',
                nbinsx=50,
                marker_color='rgba(73, 86, 195, 0.6)'
            ),
            row=1, col=1
        )
        
        # Update x-axis with actual price labels
        fig.update_xaxes(
            title_text='Price ($)',
            ticktext=[f'${10**x:,.0f}' for x in range(0, int(max(log_prices))+1)],
            tickvals=list(range(0, int(max(log_prices))+1)),
            row=1, col=1
        )
        
        # 2. Price by Brand - Violin plots
        top_brands = self.df['brand'].value_counts().head(10).index
        brand_data = self.df[self.df['brand'].isin(top_brands)]
        
        fig.add_trace(
            go.Violin(
                x=brand_data['brand'],
                y=brand_data['price_clean'],
                box_visible=True,
                meanline_visible=True,
                points='outliers',
                name='Price by Brand'
            ),
            row=1, col=2
        )
        
        # 3. Price Segments
        segment_counts = self.df['price_segment'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=segment_counts.index,
                y=segment_counts.values,
                text=segment_counts.values,
                textposition='auto',
                marker_color='rgba(50, 171, 96, 0.7)',
                name='Price Segments'
            ),
            row=2, col=1
        )
        
        # 4. Price Trend by Year with confidence interval
        yearly_stats = self.df.groupby('year').agg({
            'price_clean': ['mean', 'std', 'count']
        }).reset_index()
        yearly_stats.columns = ['year', 'mean_price', 'std_price', 'count']
        
        # Calculate confidence interval
        confidence = 0.95
        yearly_stats['ci'] = (yearly_stats['std_price'] / 
                            np.sqrt(yearly_stats['count']) * 
                            stats.t.ppf((1 + confidence) / 2, yearly_stats['count'] - 1))
        
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['mean_price'],
                mode='lines',
                name='Mean Price',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['mean_price'] + yearly_stats['ci'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['mean_price'] - yearly_stats['ci'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(128, 0, 128, 0.2)',
                fill='tonexty',
                name='95% Confidence Interval'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Watch Price Analysis Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Brand", row=1, col=2, tickangle=45)
        fig.update_xaxes(title_text="Price Segment", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=2, type='log')
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Average Price ($)", row=2, col=2)
        
        # Save visualization
        fig.write_html(os.path.join(self.viz_dirs['price'], 'price_analysis.html'))

    def _visualize_brand_distribution(self):
        """
        Includes market share, brand relationships, and positioning.
        """
        # 1. Brand Market Share Treemap
        brand_counts = self.df['brand'].value_counts()
        fig = go.Figure(go.Treemap(
            labels=brand_counts.index,
            parents=[""] * len(brand_counts),
            values=brand_counts.values,
            textinfo="label+value+percent parent"
        ))
        fig.update_layout(title="Brand Market Share")
        fig.write_html(os.path.join(self.viz_dirs['brand'], 'brand_market_share.html'))
        
        # 2. Brand Price Positioning
        fig = px.scatter(
            self.df,
            x='year',
            y='price_clean',
            color='brand',
            size='size_clean',
            hover_data=['model'],
            title="Brand Positioning by Price and Year"
        )
        fig.write_html(os.path.join(self.viz_dirs['brand'], 'brand_positioning.html'))
        
        # 3. Brand Relationship Network
        self._create_brand_network()

    def _create_brand_network(self):
        """
        Create an improved network visualization of brand relationships.
        """
        # Get top brands and create color mapping
        brand_counts = self.df['brand'].value_counts()
        top_brands = brand_counts.head(15).index  # Limit to top 15 brands
        
        # Create brand metrics
        brand_metrics = {}
        for brand in top_brands:
            brand_data = self.df[self.df['brand'] == brand]
            brand_metrics[brand] = {
                'avg_price': brand_data['price_clean'].mean(),
                'market_share': len(brand_data),
                'avg_year': brand_data['year'].mean()
            }
        
        # Create network
        G = nx.Graph()
        
        # Add nodes with improved attributes
        for brand in top_brands:
            G.add_node(
                brand,
                node_type='brand',
                size=np.log2(brand_metrics[brand]['market_share']) * 10,
                price_level=brand_metrics[brand]['avg_price'],
                year=brand_metrics[brand]['avg_year']
            )
        
        # Add edges with multiple similarity factors
        for i, brand1 in enumerate(top_brands):
            for brand2 in top_brands[i+1:]:
                # Calculate multiple similarity metrics
                price_similarity = 1 / (1 + abs(np.log1p(brand_metrics[brand1]['avg_price']) - 
                                            np.log1p(brand_metrics[brand2]['avg_price'])))
                year_similarity = 1 / (1 + abs(brand_metrics[brand1]['avg_year'] - 
                                        brand_metrics[brand2]['avg_year']))
                
                # Combined similarity
                similarity = (price_similarity + year_similarity) / 2
                
                if similarity > 0.3:  # Only add significant connections
                    G.add_edge(brand1, brand2, weight=similarity)
        
        # Compute layout with more spacing
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Create visualization
        fig = go.Figure()
        
        # Add edges with varying opacity
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(
                        width=weight * 5,
                        color='rgba(180,180,180,0.5)'
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Add nodes with better visual encoding
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_texts = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_sizes.append(G.nodes[node[0]]['size'] * 2)
            node_colors.append(np.log10(G.nodes[node[0]]['price_level']))
            
            # Create detailed hover text
            text = (f"Brand: {node[0]}<br>" +
                    f"Avg Price: ${G.nodes[node[0]]['price_level']:,.2f}<br>" +
                    f"Market Share: {np.exp(G.nodes[node[0]]['size']/10):,.0f} watches<br>" +
                    f"Avg Year: {G.nodes[node[0]]['year']:.1f}")
            node_texts.append(text)
        
        # Add nodes trace
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Log Price ($)')
                ),
                text=list(G.nodes()),
                textposition="top center",
                hovertext=node_texts,
                hoverinfo='text'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Watch Brand Network<br>" +
                    "<sup>Node size: Market share | Color: Price level | Connections: Brand similarity</sup>",
                x=0.5,
                y=0.95
            ),
            showlegend=False,
            width=1200,
            height=800,
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Save visualization
        fig.write_html(os.path.join(self.viz_dirs['network'], 'brand_network.html'))

    def _visualize_feature_correlations(self):
        """
        Includes numerical correlations, categorical relationships, and feature distributions.
        """
        # 1. Numerical Feature Correlations
        numerical_cols = ['price_clean', 'size_clean', 'year']
        corr_matrix = self.df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Feature Correlations")
        fig.write_html(os.path.join(self.viz_dirs['eda'], 'feature_correlations.html'))
        
        # 2. Feature Pair Plots
        fig = make_subplots(rows=3, cols=3, subplot_titles=[
            'Price vs Size', 'Price vs Year', 'Size vs Year',
            'Price Distribution', 'Size Distribution', 'Year Distribution',
            'Brand Distribution', 'Price Segment Distribution', 'Size Segment Distribution'
        ])
        
        # Scatter plots
        fig.add_trace(
            go.Scatter(x=self.df['size_clean'], y=self.df['price_clean'],
                      mode='markers', name='Price vs Size', opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['year'], y=self.df['price_clean'],
                      mode='markers', name='Price vs Year', opacity=0.5),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.df['year'], y=self.df['size_clean'],
                      mode='markers', name='Size vs Year', opacity=0.5),
            row=1, col=3
        )
        
        # Distributions
        fig.add_trace(
            go.Histogram(x=self.df['price_clean'], name='Price Dist'),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=self.df['size_clean'], name='Size Dist'),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=self.df['year'], name='Year Dist'),
            row=2, col=3
        )
        
        # Categorical distributions
        fig.add_trace(
            go.Bar(x=self.df['brand'].value_counts().head(10).index,
                  y=self.df['brand'].value_counts().head(10).values,
                  name='Top Brands'),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=self.df['price_segment'].value_counts().index,
                  y=self.df['price_segment'].value_counts().values,
                  name='Price Segments'),
            row=3, col=2
        )
        fig.add_trace(
            go.Bar(x=self.df['size_segment'].value_counts().index,
                  y=self.df['size_segment'].value_counts().values,
                  name='Size Segments'),
            row=3, col=3
        )
        
        fig.update_layout(height=1500, width=1500, showlegend=False)
        fig.write_html(os.path.join(self.viz_dirs['eda'], 'feature_pairs.html'))
    
    def visualize_network_analysis(self):
        """
        Create improved starburst/flower-like network visualization.
        """
        print("Generating network visualizations...")
        
        try:
            # Create base network
            G = nx.Graph()
            
            # Get top brands
            brand_counts = self.df['brand'].value_counts()
            top_brands = brand_counts.head(15).index
            
            # Add central node with enhanced styling
            G.add_node("Watch Market", node_type="root", size=50)
            
            # Add brand nodes with metrics
            for brand in top_brands:
                brand_data = self.df[self.df['brand'] == brand]
                market_share = len(brand_data)
                avg_price = brand_data['price_clean'].mean()
                
                G.add_node(brand, 
                        node_type="brand",
                        size=np.log2(market_share) * 5,
                        market_share=market_share,
                        avg_price=avg_price)
                G.add_edge("Watch Market", brand, weight=market_share)
            
            # Add watch nodes (limited per brand)
            max_watches_per_brand = 15
            for brand in top_brands:
                brand_watches = self.df[self.df['brand'] == brand].head(max_watches_per_brand)
                
                for _, watch in brand_watches.iterrows():
                    watch_id = f"{watch['brand']}_{watch['model']}"
                    G.add_node(watch_id,
                            node_type="watch",
                            size=5,
                            price=float(watch['price_clean']))
                    G.add_edge(brand, watch_id)
            
            # Create starburst layout
            pos = nx.spring_layout(G, k=5, iterations=50)
            
            # Adjust positions for starburst effect
            center_pos = pos["Watch Market"]
            for node in G.nodes():
                if node != "Watch Market":
                    if G.nodes[node]["node_type"] == "brand":
                        angle = 2 * np.pi * list(top_brands).index(node) / len(top_brands)
                        r = 0.3
                        pos[node] = np.array([
                            center_pos[0] + r * np.cos(angle),
                            center_pos[1] + r * np.sin(angle)
                        ])
                    else:
                        brand = node.split('_')[0]
                        base_angle = 2 * np.pi * list(top_brands).index(brand) / len(top_brands)
                        angle = base_angle + np.random.normal(0, 0.1)
                        r = 0.6 + np.random.normal(0, 0.05)
                        pos[node] = np.array([
                            center_pos[0] + r * np.cos(angle),
                            center_pos[1] + r * np.sin(angle)
                        ])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            edge_traces = []
            
            # Brand connections (green)
            brand_x = []
            brand_y = []
            for edge in G.edges():
                if G.nodes[edge[0]]["node_type"] == "root" or G.nodes[edge[1]]["node_type"] == "root":
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    brand_x.extend([x0, x1, None])
                    brand_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(x=brand_x, y=brand_y,
                        line=dict(width=2, color='#00ff00'),
                        hoverinfo='none',
                        mode='lines',
                        name='Brand connections')
            )
            
            # Watch connections (blue)
            watch_x = []
            watch_y = []
            for edge in G.edges():
                if G.nodes[edge[1]]["node_type"] == "watch":
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    watch_x.extend([x0, x1, None])
                    watch_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(x=watch_x, y=watch_y,
                        line=dict(width=1, color='#0000ff'),
                        hoverinfo='none',
                        mode='lines',
                        name='Watch connections')
            )
            
            # Add nodes
            for node_type, color, size in [("root", '#ffffff', 30),
                                        ("brand", '#00ff00', 20),
                                        ("watch", '#0000ff', 5)]:
                node_x = []
                node_y = []
                node_text = []
                hover_text = []
                
                for node in G.nodes():
                    if G.nodes[node]["node_type"] == node_type:
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        if node_type == "root":
                            text = "Watch Market"
                            hover = "Global Watch Market"
                        elif node_type == "brand":
                            text = node
                            hover = f"Brand: {node}<br>Market Share: {G.nodes[node]['market_share']}"
                        else:
                            text = ""
                            hover = f"Model: {node.split('_')[1]}<br>Price: ${G.nodes[node]['price']:,.2f}"
                        
                        node_text.append(text)
                        hover_text.append(hover)
                
                fig.add_trace(
                    go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            marker=dict(size=size, color=color),
                            text=node_text,
                            textposition="middle center",
                            hovertext=hover_text,
                            hoverinfo='text',
                            name=node_type.capitalize())
                )
            
            # Update layout
            fig.update_layout(
                title="Watch Market Network - Hierarchical View",
                showlegend=True,
                width=1200,
                height=1200,
                template='plotly_dark',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Save visualization
            fig.write_html(os.path.join(self.viz_dirs['network'], 'starburst_network.html'))
            print("Network visualization completed successfully")
            
        except Exception as e:
            print(f"Error in network visualization: {str(e)}")
            raise

    def _save_network_stats(self, stats):
        """Save network statistics to a file"""
        stats_path = os.path.join(self.viz_dirs['network'], 'network_stats.html')
        
        html_content = f"""
        <html>
        <head>
            <title>Network Statistics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .stat-card {{ 
                    background: #f5f5f5;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                    display: inline-block;
                    min-width: 200px;
                }}
                .stat-value {{ 
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
            </style>
        </head>
        <body>
            <h1>Network Statistics</h1>
            <div class="stat-card">
                <h3>Total Nodes</h3>
                <div class="stat-value">{stats['nodes']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Total Edges</h3>
                <div class="stat-value">{stats['edges']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Network Density</h3>
                <div class="stat-value">{stats['density']:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Average Degree</h3>
                <div class="stat-value">{stats['avg_degree']:.2f}</div>
            </div>
        </body>
        </html>
        """
        
        with open(stats_path, 'w') as f:
            f.write(html_content)

    def _visualize_centrality_measures(self, G, pos, degree_cent, between_cent, eigen_cent):
        """
        Create visualizations for different centrality measures.
        Optimized for speed and memory usage.
        """
        print("Creating centrality measure plots...")
        
        fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=['Degree Centrality', 
                                        'Betweenness Centrality', 
                                        'Eigenvector/Degree Centrality'])
        
        def create_network_trace(centrality_dict, row, col):
            node_x = []
            node_y = []
            node_color = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                centrality = centrality_dict.get(node, 0)
                node_color.append(centrality)
                node_text.append(f'Node: {node}<br>Centrality: {centrality:.3f}')
            
            return go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=8,  
                    color=node_color,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=node_text,
                hoverinfo='text',
                showlegend=False
            )
        
        # Add traces for each centrality measure
        fig.add_trace(create_network_trace(degree_cent, 1, 1), row=1, col=1)
        fig.add_trace(create_network_trace(between_cent, 1, 2), row=1, col=2)
        fig.add_trace(create_network_trace(eigen_cent, 1, 3), row=1, col=3)
        
        fig.update_layout(
            height=600, 
            width=1800, 
            title_text="Network Centrality Measures",
            showlegend=False,
            hovermode='closest'
        )
        
        print("Saving centrality visualization...")
        fig.write_html(os.path.join(self.viz_dirs['network'], 'centrality_measures.html'))

    def _visualize_centrality_measures(self, G, pos, degree_cent, between_cent, eigen_cent):
        """
        Create visualizations for different centrality measures.
        Now handles potential None values for centrality measures.
        """
        fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=['Degree Centrality', 
                                        'Betweenness Centrality', 
                                        'Eigenvector/Degree Centrality'])
        
        # Helper function for creating network traces
        def create_network_trace(centrality_dict, row, col):
            node_x = []
            node_y = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_color.append(centrality_dict.get(node, 0))  
            
            return go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=10,
                    color=node_color,
                    colorscale='Viridis',
                    showscale=True
                ),
                showlegend=False
            )
        
        # Add traces for each centrality measure
        fig.add_trace(create_network_trace(degree_cent, 1, 1), row=1, col=1)
        fig.add_trace(create_network_trace(between_cent, 1, 2), row=1, col=2)
        fig.add_trace(create_network_trace(eigen_cent, 1, 3), row=1, col=3)
        
        fig.update_layout(height=600, width=1800, title_text="Network Centrality Measures")
        fig.write_html(os.path.join(self.viz_dirs['network'], 'centrality_measures.html'))

    def _visualize_communities(self, G, pos, communities):
        """
        Create visualization for community structure in the network.
        """
        # Create edges trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes trace
        node_x = []
        node_y = []
        node_color = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(communities[node])
            
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=10,
                color=node_color,
                colorscale='Viridis',
                showscale=True
            ),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[edges_trace, nodes_trace],
                     layout=go.Layout(
                         title='Network Communities',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        fig.write_html(os.path.join(self.viz_dirs['network'], 'communities.html'))
    def _visualize_degree_distribution(self, G):
        """
        Create visualization for node degree distribution and related network metrics.
        """
        degrees = [d for n, d in G.degree()]
        
        # Create subplot figure
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Degree Distribution',
                                         'Log-Log Degree Distribution',
                                         'Cumulative Degree Distribution',
                                         'Local Clustering Coefficient'])
        
        # Regular degree distribution
        fig.add_trace(
            go.Histogram(x=degrees, name='Degree Dist'),
            row=1, col=1
        )
        
        # Log-log degree distribution
        degree_count = pd.Series(degrees).value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=np.log10(degree_count.index),
                      y=np.log10(degree_count.values),
                      mode='markers',
                      name='Log-Log Degree'),
            row=1, col=2
        )
        
        # Cumulative degree distribution
        sorted_degrees = np.sort(degrees)
        cum_dist = 1. * np.arange(len(sorted_degrees)) / (len(sorted_degrees) - 1)
        fig.add_trace(
            go.Scatter(x=sorted_degrees, y=cum_dist,
                      mode='lines',
                      name='Cumulative Dist'),
            row=2, col=1
        )
        
        # Local clustering coefficients
        clustering_coef = nx.clustering(G)
        fig.add_trace(
            go.Histogram(x=list(clustering_coef.values()),
                        name='Clustering Coef'),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, width=1200,
                         title_text="Network Degree Analysis")
        fig.write_html(os.path.join(self.viz_dirs['network'], 'degree_analysis.html'))

    def visualize_embedding_space(self):
        """
        Create comprehensive visualizations of the embedding space with NaN handling.
        """
        print("Generating embedding space visualizations...")
        
        try:
            # Sample data for visualization
            sample_size = min(50000, len(self.embeddings))
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            sample_embeddings = self.embeddings[indices].copy()  # Make a copy
            sample_df = self.df.iloc[indices].copy()
            
            # More thorough NaN checking
            print("Performing thorough data validation...")
            
            # Check for and remove NaN values in embeddings
            nan_mask_embeddings = ~np.isnan(sample_embeddings).any(axis=1)
            inf_mask_embeddings = ~np.isinf(sample_embeddings).any(axis=1)
            
            # Check for NaN values in relevant DataFrame columns
            nan_mask_df = ~pd.isna(sample_df[['price_clean', 'year', 'brand']]).any(axis=1)
            
            # Combine all masks
            valid_mask = nan_mask_embeddings & inf_mask_embeddings & nan_mask_df
            
            if not np.all(valid_mask):
                print(f"Found {np.sum(~valid_mask)} invalid samples, removing them...")
                sample_embeddings = sample_embeddings[valid_mask]
                sample_df = sample_df[valid_mask].reset_index(drop=True)
            
            # Additional validation
            if len(sample_embeddings) == 0:
                raise ValueError("No valid samples remaining after filtering")
            
            print(f"Processing {len(sample_embeddings)} valid samples...")
            
            # Robust standardization
            def robust_standardize(data):
                """Standardize data with outlier handling"""
                median = np.median(data, axis=0)
                mad = np.median(np.abs(data - median), axis=0)
                mad[mad == 0] = 1  # Prevent division by zero
                return (data - median) / mad
            
            print("Performing robust standardization...")
            sample_embeddings_standardized = robust_standardize(sample_embeddings)
            
            # Verify standardization worked
            if np.any(np.isnan(sample_embeddings_standardized)):
                print("Warning: NaN values after standardization, using original embeddings")
                sample_embeddings_standardized = sample_embeddings
            
            # Clip extreme values
            print("Clipping extreme values...")
            sample_embeddings_standardized = np.clip(
                sample_embeddings_standardized, 
                -10, 
                10
            )
            
            # 1. UMAP Visualization
            print("Generating UMAP visualization...")
            try:
                self._visualize_umap_embeddings(sample_embeddings_standardized, sample_df)
                print("UMAP visualization completed")
            except Exception as e:
                print(f"Error in UMAP visualization: {str(e)}")
            
            # 2. t-SNE Visualization
            print("Generating t-SNE visualization...")
            try:
                self._visualize_tsne_embeddings(sample_embeddings_standardized, sample_df)
                print("t-SNE visualization completed")
            except Exception as e:
                print(f"Error in t-SNE visualization: {str(e)}")
            
            # 3. PCA Visualization (as a fallback)
            print("Generating PCA visualization...")
            try:
                self._visualize_pca_embeddings(sample_embeddings_standardized, sample_df)
                print("PCA visualization completed")
            except Exception as e:
                print(f"Error in PCA visualization: {str(e)}")
            
            print("Embedding visualizations completed")
            
        except Exception as e:
            print(f"Error in embedding visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _visualize_pca_embeddings(self, embeddings, df):
        """
        Create PCA visualization with better interpretability.
        """
        # Compute PCA
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(embeddings)
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        total_var = sum(explained_var)
        
        # Create figure
        fig = go.Figure()
        
        # Apply log scaling to prices
        log_prices = np.log1p(df['price_clean'])
        
        # Create hover text
        hover_text = df.apply(
            lambda x: f"Brand: {x['brand']}<br>" +
                    f"Model: {x['model']}<br>" +
                    f"Price: ${x['price_clean']:,.2f}<br>" +
                    f"Year: {x['year']}<br>" +
                    f"Size: {x['size_clean']}mm",
            axis=1
        )
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=log_prices,
                    colorscale='Viridis',
                    colorbar=dict(
                        title='Log Price ($)',
                        tickformat='$,.0f',
                        tickvals=np.linspace(log_prices.min(), log_prices.max(), 6),
                        ticktext=[f'${np.exp(val):,.0f}' for val in np.linspace(log_prices.min(), log_prices.max(), 6)]
                    ),
                    showscale=True
                ),
                text=hover_text,
                hoverinfo='text'
            )
        )
        
        # Add brand labels for major clusters
        brands = df['brand'].value_counts().head(8).index
        for brand in brands:
            brand_mask = df['brand'] == brand
            if sum(brand_mask) > 0:
                x_mean = embedding_2d[brand_mask, 0].mean()
                y_mean = embedding_2d[brand_mask, 1].mean()
                
                fig.add_annotation(
                    x=x_mean,
                    y=y_mean,
                    text=brand,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    font=dict(size=14, color='black'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#636363',
                    borderwidth=1
                )
        
        # Add explained variance annotations
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"PC1 Explained Variance: {explained_var[0]:.1%}<br>" +
                f"PC2 Explained Variance: {explained_var[1]:.1%}<br>" +
                f"Total Explained Variance: {total_var:.1%}",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor='rgba(255,255,255,0.8)'
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Watch Embedding Space (PCA)<br>" +
                    "<sup>Colors: Log-scaled prices | PC1 & PC2: Main variation directions in data</sup>",
                x=0.5,
                y=0.95
            ),
            width=1200,
            height=800,
            template='plotly_white',
            xaxis_title="First Principal Component (PC1)",
            yaxis_title="Second Principal Component (PC2)"
        )
        
        # Save the visualization
        output_path = os.path.join(self.viz_dirs['embedding'], 'pca_visualization.html')
        fig.write_html(output_path)
        print(f"Saved PCA visualization to {output_path}")
        
        return fig

    def _visualize_umap_embeddings(self, embeddings, df):
        """
        Create UMAP visualization with better color scaling and interpretability.
        """
        # Configure UMAP with better parameters
        reducer = umap.UMAP(
            n_neighbors=30,        
            min_dist=0.1,        
            n_components=2,
            random_state=42,
            metric='euclidean'
        )
        
        # Compute UMAP projection
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create figure
        fig = go.Figure()
        
        # Apply log scaling to prices for better color distribution
        log_prices = np.log1p(df['price_clean'])  # log1p to handle zero values
        
        # Add scatter plot with hover information
        fig.add_trace(
            go.Scatter(
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=log_prices,
                    colorscale='Viridis',
                    colorbar=dict(
                        title='Log Price ($)',
                        tickformat='$,.0f',
                        tickvals=np.linspace(log_prices.min(), log_prices.max(), 6),
                        ticktext=[f'${np.exp(val):,.0f}' for val in np.linspace(log_prices.min(), log_prices.max(), 6)]
                    ),
                    showscale=True
                ),
                text=df.apply(
                    lambda x: f"Brand: {x['brand']}<br>" +
                            f"Model: {x['model']}<br>" +
                            f"Price: ${x['price_clean']:,.2f}<br>" +
                            f"Year: {x['year']}<br>" +
                            f"Size: {x['size_clean']}mm",
                    axis=1
                ),
                hoverinfo='text'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Watch Embedding Space (UMAP)<br>" +
                    "<sup>Colors: Log-scaled prices | Distance: Similarity between watches</sup>",
                x=0.5,
                y=0.95
            ),
            width=1200,
            height=800,
            template='plotly_white',
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2"
        )
        
        # Add annotations for major clusters
        brands = df['brand'].value_counts().head(5).index
        for brand in brands:
            brand_mask = df['brand'] == brand
            if sum(brand_mask) > 0:
                x_mean = embedding_2d[brand_mask, 0].mean()
                y_mean = embedding_2d[brand_mask, 1].mean()
                
                fig.add_annotation(
                    x=x_mean,
                    y=y_mean,
                    text=brand,
                    showarrow=False,
                    font=dict(size=14, color='black'),
                    bgcolor='rgba(255,255,255,0.8)'
                )
        
        # Add the save functionality at the end of the function, before the return
        output_path = os.path.join(self.viz_dirs['embedding'], 'umap_visualization.html')
        fig.write_html(output_path)
        
        return fig
    def _visualize_tsne_embeddings(self, embeddings, df):
        """
        Create improved t-SNE visualization with better color scaling and interpretability.
        """
        # Configure t-SNE with better parameters
        tsne = TSNE(
            n_components=2,
            perplexity=50,         
            early_exaggeration=12,  
            random_state=42,
            n_iter=1000            
        )
        
        # Compute t-SNE projection
        embedding_2d = tsne.fit_transform(embeddings)
        
        # Create figure
        fig = go.Figure()
        
        # Apply log scaling to prices for better color distribution
        log_prices = np.log1p(df['price_clean'])
        
        # Create hover text
        hover_text = df.apply(
            lambda x: f"Brand: {x['brand']}<br>" +
                    f"Model: {x['model']}<br>" +
                    f"Price: ${x['price_clean']:,.2f}<br>" +
                    f"Year: {x['year']}<br>" +
                    f"Size: {x['size_clean']}mm",
            axis=1
        )
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=log_prices,
                    colorscale='Viridis',
                    colorbar=dict(
                        title='Log Price ($)',
                        tickformat='$,.0f',
                        tickvals=np.linspace(log_prices.min(), log_prices.max(), 6),
                        ticktext=[f'${np.exp(val):,.0f}' for val in np.linspace(log_prices.min(), log_prices.max(), 6)]
                    ),
                    showscale=True
                ),
                text=hover_text,
                hoverinfo='text'
            )
        )
        
        # Add brand labels for major clusters
        brands = df['brand'].value_counts().head(8).index  
        for brand in brands:
            brand_mask = df['brand'] == brand
            if sum(brand_mask) > 0:
                x_mean = embedding_2d[brand_mask, 0].mean()
                y_mean = embedding_2d[brand_mask, 1].mean()
                
                # Calculate cluster density to place label in densest area
                from scipy.stats import gaussian_kde
                points = embedding_2d[brand_mask]
                if len(points) > 1:  
                    kde = gaussian_kde(points.T)
                    density = kde(points.T)
                    max_density_idx = np.argmax(density)
                    x_mean = points[max_density_idx, 0]
                    y_mean = points[max_density_idx, 1]
                
                fig.add_annotation(
                    x=x_mean,
                    y=y_mean,
                    text=f"{brand}",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    font=dict(size=14, color='black'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#636363',
                    borderwidth=1
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Watch Embedding Space (t-SNE)<br>" +
                    "<sup>Colors: Log-scaled prices | Distance: Similarity between watches</sup>",
                x=0.5,
                y=0.95
            ),
            width=1200,
            height=800,
            template='plotly_white',
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            showlegend=False
        )
        
        # Add a note about the visualization
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text="Clusters indicate similar watches<br>based on all features",
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
        
        # Save the visualization
        output_path = os.path.join(self.viz_dirs['embedding'], 'tsne_visualization.html')
        fig.write_html(output_path)
        print(f"Saved t-SNE visualization to {output_path}")
        
        return fig

    def _visualize_embedding_clusters(self, embeddings, df):
        """
        Create visualization of embedding clusters with interactive features.
        """
        try:
            print("Computing clusters...")
            
            # Standardize the embeddings
            embeddings_scaled = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
            
            # Perform clustering
            n_clusters = 8 
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings_scaled)
            
            print("Computing UMAP projection for cluster visualization...")
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                metric='euclidean',
                low_memory=True
            )
            embedding_2d = reducer.fit_transform(embeddings_scaled)
            
            # Create cluster visualization
            fig = go.Figure()
            
            # Add scatter plot for each cluster
            for i in range(n_clusters):
                mask = clusters == i
                hover_text = df[mask].apply(
                    lambda x: f"Cluster: {i}<br>Brand: {x['brand']}<br>Model: {x['model']}<br>Price: ${x['price_clean']:,.2f}",
                    axis=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=embedding_2d[mask, 0],
                        y=embedding_2d[mask, 1],
                        mode='markers',
                        name=f'Cluster {i}',
                        text=hover_text,
                        hoverinfo='text',
                        marker=dict(size=5)
                    )
                )
            
            fig.update_layout(
                title="Embedding Clusters",
                width=1200,
                height=800,
                template='plotly_white',
                showlegend=True
            )
            
            output_path = os.path.join(self.viz_dirs['embedding'], 'clusters.html')
            fig.write_html(output_path)
            print("Saved cluster visualization")
            
        except Exception as e:
            print(f"Error in cluster visualization: {str(e)}")
            raise
    
    def visualize_temporal_analysis(self):
        """
        Create comprehensive temporal analysis visualizations.
        Includes trends, patterns, and temporal relationships.
        """
        print("Generating temporal analysis visualizations...")
        
        # 1. Year Distribution and Trends
        self._visualize_year_distribution()
        
        # 2. Price Evolution Over Time
        self._visualize_price_evolution()
        
        # 3. Brand Evolution
        self._visualize_brand_evolution()
        
        # 4. Market Trends
        self._visualize_market_trends()
        
        print("Temporal analysis completed.")

    def _visualize_year_distribution(self):
        """
        Create detailed visualization of watch distribution across years.
        """
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Year Distribution',
                                         'Decade Distribution',
                                         'Cumulative Production',
                                         'Year vs Price'])
        
        # Year distribution
        fig.add_trace(
            go.Histogram(x=self.df['year'],
                        name='Year Distribution',
                        nbinsx=50),
            row=1, col=1
        )
        
        # Decade distribution
        decade_counts = self.df['decade'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=decade_counts.index,
                  y=decade_counts.values,
                  name='Decade Distribution'),
            row=1, col=2
        )
        
        # Cumulative production
        year_counts = self.df['year'].value_counts().sort_index().cumsum()
        fig.add_trace(
            go.Scatter(x=year_counts.index,
                      y=year_counts.values,
                      name='Cumulative Production'),
            row=2, col=1
        )
        
        # Year vs Price
        fig.add_trace(
            go.Scatter(x=self.df['year'],
                      y=self.df['price_clean'],
                      mode='markers',
                      opacity=0.5,
                      name='Year vs Price'),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, width=1200,
                         title_text="Temporal Distribution Analysis")
        fig.write_html(os.path.join(self.viz_dirs['temporal'], 'year_distribution.html'))

    def _visualize_price_evolution(self):
        """
        Create visualization of price evolution over time with multiple perspectives.
        """
        # Group data by year
        yearly_stats = self.df.groupby('year').agg({
            'price_clean': ['mean', 'median', 'std', 'count']
        }).reset_index()
        yearly_stats.columns = ['year', 'mean_price', 'median_price', 'std_price', 'count']
        
        fig = go.Figure()
        
        # Add mean price line
        fig.add_trace(
            go.Scatter(x=yearly_stats['year'],
                      y=yearly_stats['mean_price'],
                      name='Mean Price',
                      line=dict(color='blue'))
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(x=yearly_stats['year'],
                      y=yearly_stats['mean_price'] + yearly_stats['std_price'],
                      fill=None,
                      mode='lines',
                      line_color='rgba(0,100,255,0.2)',
                      showlegend=False)
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_stats['year'],
                      y=yearly_stats['mean_price'] - yearly_stats['std_price'],
                      fill='tonexty',
                      mode='lines',
                      line_color='rgba(0,100,255,0.2)',
                      name='Price Range')
        )
        
        # Add median price line
        fig.add_trace(
            go.Scatter(x=yearly_stats['year'],
                      y=yearly_stats['median_price'],
                      name='Median Price',
                      line=dict(color='red', dash='dash'))
        )
        
        # Add volume indicator
        fig.add_trace(
            go.Bar(x=yearly_stats['year'],
                  y=yearly_stats['count'],
                  yaxis='y2',
                  name='Number of Watches',
                  opacity=0.3)
        )
        
        fig.update_layout(
            title="Price Evolution Over Time",
            xaxis_title="Year",
            yaxis_title="Price",
            yaxis2=dict(
                title="Number of Watches",
                overlaying='y',
                side='right'
            ),
            height=600,
            width=1200
        )
        
        fig.write_html(os.path.join(self.viz_dirs['temporal'], 'price_evolution.html'))

    def _visualize_brand_evolution(self):
        """
        Create visualization of brand evolution over time.
        """
        # Get top brands
        top_brands = self.df['brand'].value_counts().head(10).index
        brand_data = self.df[self.df['brand'].isin(top_brands)]
        
        # Create brand evolution heatmap
        brand_year_counts = pd.crosstab(brand_data['year'], brand_data['brand'])
        brand_year_counts = brand_year_counts.fillna(0)
        
        # Normalize by year
        brand_year_pct = brand_year_counts.div(brand_year_counts.sum(axis=1), axis=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=brand_year_pct.values,
            x=brand_year_pct.columns,
            y=brand_year_pct.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Brand Market Share Evolution",
            xaxis_title="Brand",
            yaxis_title="Year",
            height=800,
            width=1200
        )
        
        fig.write_html(os.path.join(self.viz_dirs['temporal'], 'brand_evolution.html'))
        
        # Create brand price evolution
        fig = go.Figure()
        
        for brand in top_brands:
            brand_yearly_price = brand_data[brand_data['brand'] == brand].groupby('year')['price_clean'].mean()
            fig.add_trace(
                go.Scatter(x=brand_yearly_price.index,
                          y=brand_yearly_price.values,
                          name=brand,
                          mode='lines+markers')
            )
        
        fig.update_layout(
            title="Brand Price Evolution",
            xaxis_title="Year",
            yaxis_title="Average Price",
            height=600,
            width=1200
        )
        
        fig.write_html(os.path.join(self.viz_dirs['temporal'], 'brand_price_evolution.html'))

    def _visualize_market_trends(self):
        """
        Create visualization of overall market trends and patterns.
        """
        # Create subplots for different trends
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Size Evolution',
                                         'Price Segment Evolution',
                                         'Material Trends',
                                         'Condition Distribution'])
        
        # Size evolution
        yearly_size = self.df.groupby('year')['size_clean'].mean()
        fig.add_trace(
            go.Scatter(x=yearly_size.index,
                      y=yearly_size.values,
                      name='Average Size'),
            row=1, col=1
        )
        
        # Price segment evolution
        segment_year = pd.crosstab(self.df['year'], self.df['price_segment'], normalize='index')
        for segment in segment_year.columns:
            fig.add_trace(
                go.Scatter(x=segment_year.index,
                          y=segment_year[segment],
                          name=segment,
                          stackgroup='one'),
                row=1, col=2
            )
        
        # Material trends
        material_year = pd.crosstab(self.df['year'], self.df['casem'])
        material_year_pct = material_year.div(material_year.sum(axis=1), axis=0)
        for material in material_year_pct.columns[:5]:  # Top 5 materials
            fig.add_trace(
                go.Scatter(x=material_year_pct.index,
                          y=material_year_pct[material],
                          name=material),
                row=2, col=1
            )
        
        # Condition distribution over time
        condition_year = pd.crosstab(self.df['year'], self.df['condition'], normalize='index')
        for condition in condition_year.columns:
            fig.add_trace(
                go.Scatter(x=condition_year.index,
                          y=condition_year[condition],
                          name=condition,
                          stackgroup='one'),
                row=2, col=2
            )
        
        fig.update_layout(height=1000, width=1200,
                         title_text="Market Trends Analysis")
        fig.write_html(os.path.join(self.viz_dirs['temporal'], 'market_trends.html'))

    def find_similar_watches(self, watch_idx, n_similar=10):
        """
        Find similar watches based on embedding distance with robust NaN handling.
        """
        # Get the target watch embedding
        target_embedding = self.embeddings[watch_idx]
        
        # Check if target embedding has NaN values
        if np.any(np.isnan(target_embedding)):
            raise ValueError("Target watch embedding contains NaN values")
        
        # Create a mask for valid embeddings (no NaN values)
        valid_mask = ~np.any(np.isnan(self.embeddings), axis=1)
        valid_embeddings = self.embeddings[valid_mask]
        
        # Calculate cosine similarity with all valid embeddings
        similarities = cosine_similarity([target_embedding], valid_embeddings)[0]
        
        # Get the indices in the original dataset
        valid_indices = np.where(valid_mask)[0]
        
        # Get indices of most similar watches (excluding self if present)
        similar_mask = valid_indices != watch_idx
        sorted_indices = np.argsort(similarities[similar_mask])[::-1][:n_similar]
        similar_indices = valid_indices[similar_mask][sorted_indices]
        
        # Create DataFrame with similar watches
        similar_watches = self.df.iloc[similar_indices].copy()
        similar_watches['similarity_score'] = similarities[similar_mask][sorted_indices]
        
        return similar_watches

    def create_interactive_similarity_search(self):
        """
        Create interactive similarity search visualizations with error handling.
        """
        print("Generating similarity search visualizations...")
        
        try:
            # Create a mask for valid embeddings
            valid_mask = ~np.any(np.isnan(self.embeddings), axis=1)
            
            # Get indices of valid watches sorted by price
            valid_df = self.df[valid_mask].copy()
            valid_embeddings = self.embeddings[valid_mask]
            
            # Get top watches by price that have valid embeddings
            top_watches = valid_df.nlargest(5, 'price_clean')
            
            for idx, watch in top_watches.iterrows():
                try:
                    print(f"Processing similar watches for {watch['brand']} {watch['model']}...")
                    
                    # Get the index in the original dataset
                    original_idx = valid_df.index.get_loc(idx)
                    similar_watches = self.find_similar_watches(original_idx, n_similar=10)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add scatter plot of similar watches
                    fig.add_trace(
                        go.Scatter(
                            x=similar_watches['price_clean'],
                            y=similar_watches['similarity_score'],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=similar_watches['similarity_score'],
                                colorscale='Viridis',
                                showscale=True
                            ),
                            text=similar_watches.apply(
                                lambda x: f"Brand: {x['brand']}<br>" +
                                        f"Model: {x['model']}<br>" +
                                        f"Price: ${x['price_clean']:,.2f}<br>" +
                                        f"Similarity: {x['similarity_score']:.3f}",
                                axis=1
                            ),
                            hoverinfo='text'
                        )
                    )
                    
                    # Add reference watch
                    fig.add_trace(
                        go.Scatter(
                            x=[watch['price_clean']],
                            y=[1.0],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='star'
                            ),
                            name='Reference Watch',
                            text=f"Reference Watch<br>" +
                                f"Brand: {watch['brand']}<br>" +
                                f"Model: {watch['model']}<br>" +
                                f"Price: ${watch['price_clean']:,.2f}",
                            hoverinfo='text'
                        )
                    )
                    
                    fig.update_layout(
                        title=f"Similar Watches to {watch['brand']} {watch['model']}",
                        xaxis_title="Price ($)",
                        yaxis_title="Similarity Score",
                        width=1000,
                        height=600,
                        showlegend=True,
                        template='plotly_white'
                    )
                    
                    # Save visualization
                    output_path = os.path.join(
                        self.viz_dirs['similarity'], 
                        f"similar_watches_{original_idx}.html"
                    )
                    fig.write_html(output_path)
                    print(f"Saved similarity visualization for watch {original_idx}")
                    
                except Exception as e:
                    print(f"Error processing watch {idx}: {str(e)}")
                    continue
            
            # Create summary visualization
            self._create_similarity_summary()
            
            print("Similarity search visualizations completed successfully")
            
        except Exception as e:
            print(f"Error in similarity search visualization: {str(e)}")
            raise

    def _create_similarity_summary(self):
        """
        Create a summary visualization of similarity patterns with NaN handling.
        """
        try:
            print("Creating similarity summary visualization...")
            
            # Create a mask for valid embeddings
            valid_mask = ~np.any(np.isnan(self.embeddings), axis=1)
            valid_df = self.df[valid_mask].copy()
            valid_embeddings = self.embeddings[valid_mask]
            
            # Sample watches for analysis from valid embeddings
            n_samples = min(1000, len(valid_df))
            sample_indices = np.random.choice(len(valid_df), n_samples, replace=False)
            
            # Calculate average similarity by brand
            brand_similarities = {}
            
            for idx in sample_indices:
                try:
                    similar_watches = self.find_similar_watches(idx, n_similar=5)
                    brand = valid_df.iloc[idx]['brand']
                    if brand not in brand_similarities:
                        brand_similarities[brand] = []
                    brand_similarities[brand].extend(similar_watches['similarity_score'].tolist())
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    continue
            
            # Calculate average similarity for each brand
            brand_avg_similarity = {
                brand: np.mean(scores) 
                for brand, scores in brand_similarities.items() 
                if len(scores) >= 5  # Only include brands with enough samples
            }
            
            # Create visualization
            fig = go.Figure()
            
            # Add bar plot of average similarities
            brands = list(brand_avg_similarity.keys())
            similarities = list(brand_avg_similarity.values())
            
            fig.add_trace(
                go.Bar(
                    x=brands,
                    y=similarities,
                    marker_color=similarities,
                    colorscale='Viridis'
                )
            )
            
            fig.update_layout(
                title="Average Similarity Scores by Brand",
                xaxis_title="Brand",
                yaxis_title="Average Similarity Score",
                width=1200,
                height=600,
                template='plotly_white'
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            # Save visualization
            output_path = os.path.join(
                self.viz_dirs['similarity'], 
                "similarity_summary.html"
            )
            fig.write_html(output_path)
            print("Saved similarity summary visualization")
            
        except Exception as e:
            print(f"Error creating similarity summary: {str(e)}")

    def generate_final_report(self):
        """
        Generate a comprehensive final report with all analyses and visualizations.
        """
        report_path = os.path.join(self.output_dir, 'final_report.html')
        
        with open(report_path, 'w') as f:
            f.write("""
            <html>
            <head>
                <title>Watch Market Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .section { margin-bottom: 30px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #34495e; }
                </style>
            </head>
            <body>
            """)
            
            # Write summary statistics
            f.write("<h1>Watch Market Analysis Report</h1>")
            f.write("<div class='section'>")
            f.write("<h2>Summary Statistics</h2>")
            f.write(f"<p>Total Watches: {len(self.df):,}</p>")
            f.write(f"<p>Total Brands: {self.df['brand'].nunique():,}</p>")
            f.write(f"<p>Price Range: ${self.df['price_clean'].min():,.2f} - ${self.df['price_clean'].max():,.2f}</p>")
            f.write(f"<p>Year Range: {int(self.df['year'].min())} - {int(self.df['year'].max())}</p>")
            f.write("</div>")
            
            # Add links to all visualizations
            for viz_type, viz_dir in self.viz_dirs.items():
                f.write(f"<div class='section'>")
                f.write(f"<h2>{viz_type.title()} Analysis</h2>")
                for viz_file in os.listdir(viz_dir):
                    if viz_file.endswith('.html'):
                        rel_path = os.path.join(os.path.basename(viz_dir), viz_file)
                        f.write(f"<p><a href='{rel_path}'>{viz_file[:-5].replace('_', ' ').title()}</a></p>")
                f.write("</div>")
            
            f.write("</body></html>")

    def visualize_force_network(self):
        """
        Create force-directed network visualization with colored nodes and thin edges.
        """
        print("Generating force-directed network visualization...")
        
        try:
            # Create base network
            G = nx.Graph()
            
            # Select top brands and create color mapping
            print("Setting up brand colors...")
            brand_counts = self.df['brand'].value_counts()
            top_brands = brand_counts.head(40).index.tolist()
            
            # Create color palette
            n_colors = len(top_brands)
            colors = px.colors.qualitative.Set3 * (n_colors // len(px.colors.qualitative.Set3) + 1)
            brand_colors = dict(zip(top_brands, colors[:n_colors]))
            
            # Add nodes and edges
            print("Building network structure...")
            nodes_added = set()
            edges_added = set()
            
            for brand in top_brands:
                # Get watches for this brand
                brand_watches = self.df[self.df['brand'] == brand].head(10)
                
                for _, watch in brand_watches.iterrows():
                    node_id = f"{watch['brand']}_{watch['model']}"[:100]  # Limit length of node ID
                    
                    if node_id not in nodes_added:
                        # Add node
                        G.add_node(
                            node_id,
                            brand=watch['brand'],
                            model=watch['model'],
                            price=float(watch['price_clean']),
                            color=brand_colors[watch['brand']]
                        )
                        nodes_added.add(node_id)
            
            # Add edges based on price similarity
            print("Adding edges...")
            nodes = list(G.nodes())
            for i, node1 in enumerate(nodes):
                node1_data = G.nodes[node1]
                
                for node2 in nodes[i+1:]:
                    node2_data = G.nodes[node2]
                    
                    # Add edge if prices are similar and brands are different
                    if (node1_data['brand'] != node2_data['brand'] and 
                        abs(node1_data['price'] - node2_data['price']) / max(node1_data['price'], node2_data['price']) < 0.2):
                        edge = tuple(sorted([node1, node2]))
                        if edge not in edges_added:
                            G.add_edge(*edge)
                            edges_added.add(edge)
            
            # Compute layout
            print("Computing force-directed layout...")
            pos = nx.spring_layout(
                G,
                k=1/np.sqrt(len(G.nodes())),
                iterations=100,
                seed=42
            )
            
            # Create visualization
            print("Creating visualization...")
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes for each brand
            for brand in top_brands:
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                
                for node in G.nodes():
                    if G.nodes[node]['brand'] == brand:
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        # Create hover text
                        text = (f"Brand: {G.nodes[node]['brand']}<br>"
                            f"Model: {G.nodes[node]['model']}<br>"
                            f"Price: ${G.nodes[node]['price']:,.2f}")
                        node_text.append(text)
                        
                        # Size based on log price
                        size = np.log(G.nodes[node]['price'] + 1) * 2
                        node_size.append(size)
                
                if node_x:  # Only add trace if there are nodes for this brand
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        text=node_text,
                        name=brand,
                        marker=dict(
                            size=node_size,
                            color=brand_colors[brand],
                            line=dict(width=1, color='#ffffff')
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title="Watch Network - Force-Directed Layout",
                showlegend=True,
                width=1500,
                height=1000,
                template='plotly_white',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(
                    itemsizing='constant',
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            # Save visualization
            print("Saving visualization...")
            output_path = os.path.join(self.viz_dirs['network'], 'force_directed_network.html')
            fig.write_html(output_path)
            print("Force-directed network visualization completed successfully")
            
        except Exception as e:
            print(f"Error in force-directed network visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def run_complete_analysis(self):
        """
        Updated run_complete_analysis to include the new visualization
        """
        print("Starting complete analysis pipeline...")
        
        try:
            # Verify data files
            if not self.verify_file_structure():
                print("Warning: Some files were not found, but continuing with available data...")
            
            print("\nGenerating visualizations...")
            
            # Create all visualization directories if they don't exist
            for dir_path in self.viz_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
            
            # Run all analyses
            self.generate_eda_visualizations()
            self.visualize_network_analysis()
            self.visualize_force_network()  # Add the new visualization
            self.visualize_embedding_space()
            self.visualize_temporal_analysis()
            self.create_interactive_similarity_search()
            
            # Generate final report
            print("\nGenerating final report...")
            self.generate_final_report()
            
            print("\nAnalysis complete! Results available in:", self.output_dir)
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            print("Attempting to continue with remaining analyses...")

    def find_similar_watches(self, watch_idx, n_similar=10):
        """
        Find similar watches based on embedding distance.
        
        Parameters:
        -----------
        watch_idx : int
            Index of the target watch
        n_similar : int
            Number of similar watches to return
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing similar watches and their similarity scores
        """
        # Get the target watch embedding
        target_embedding = self.embeddings[watch_idx]
        
        # Calculate cosine similarity with all other watches
        similarities = cosine_similarity([target_embedding], self.embeddings)[0]
        
        # Get indices of most similar watches
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        # Create DataFrame with similar watches
        similar_watches = self.df.iloc[similar_indices].copy()
        similar_watches['similarity_score'] = similarities[similar_indices]
        
        return similar_watches

    def create_interactive_similarity_search(self):
        """
        Create interactive similarity search visualizations with error handling.
        """
        print("Generating similarity search visualizations...")
        
        try:
            # Get top watches by price for demonstration
            top_watches = self.df.nlargest(5, 'price_clean')
            
            for idx, watch in top_watches.iterrows():
                try:
                    print(f"Processing similar watches for {watch['brand']} {watch['model']}...")
                    similar_watches = self.find_similar_watches(idx, n_similar=10)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add scatter plot of similar watches
                    fig.add_trace(
                        go.Scatter(
                            x=similar_watches['price_clean'],
                            y=similar_watches['similarity_score'],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=similar_watches['similarity_score'],
                                colorscale='Viridis',
                                showscale=True
                            ),
                            text=similar_watches.apply(
                                lambda x: f"Brand: {x['brand']}<br>" +
                                        f"Model: {x['model']}<br>" +
                                        f"Price: ${x['price_clean']:,.2f}<br>" +
                                        f"Similarity: {x['similarity_score']:.3f}",
                                axis=1
                            ),
                            hoverinfo='text'
                        )
                    )
                    
                    # Add reference watch
                    fig.add_trace(
                        go.Scatter(
                            x=[watch['price_clean']],
                            y=[1.0],  # Self-similarity is 1.0
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='star'
                            ),
                            name='Reference Watch',
                            text=f"Reference Watch<br>" +
                                f"Brand: {watch['brand']}<br>" +
                                f"Model: {watch['model']}<br>" +
                                f"Price: ${watch['price_clean']:,.2f}",
                            hoverinfo='text'
                        )
                    )
                    
                    fig.update_layout(
                        title=f"Similar Watches to {watch['brand']} {watch['model']}",
                        xaxis_title="Price ($)",
                        yaxis_title="Similarity Score",
                        width=1000,
                        height=600,
                        showlegend=True,
                        template='plotly_white'
                    )
                    
                    # Save visualization
                    output_path = os.path.join(
                        self.viz_dirs['similarity'], 
                        f"similar_watches_{idx}.html"
                    )
                    fig.write_html(output_path)
                    print(f"Saved similarity visualization for watch {idx}")
                    
                except Exception as e:
                    print(f"Error processing watch {idx}: {str(e)}")
                    continue
            
            # Create summary visualization
            self._create_similarity_summary()
            
            print("Similarity search visualizations completed successfully")
            
        except Exception as e:
            print(f"Error in similarity search visualization: {str(e)}")
            raise

    def _create_similarity_summary(self):
        """
        Create a summary visualization of similarity patterns.
        """
        try:
            print("Creating similarity summary visualization...")
            
            # Sample watches for analysis
            n_samples = 1000
            sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
            
            # Calculate average similarity by brand
            brand_similarities = {}
            
            for idx in sample_indices:
                similar_watches = self.find_similar_watches(idx, n_similar=5)
                brand = self.df.iloc[idx]['brand']
                if brand not in brand_similarities:
                    brand_similarities[brand] = []
                brand_similarities[brand].extend(similar_watches['similarity_score'].tolist())
            
            # Calculate average similarity for each brand
            brand_avg_similarity = {
                brand: np.mean(scores) 
                for brand, scores in brand_similarities.items() 
                if len(scores) >= 5  # Only include brands with enough samples
            }
            
            # Create visualization
            fig = go.Figure()
            
            # Add bar plot of average similarities
            brands = list(brand_avg_similarity.keys())
            similarities = list(brand_avg_similarity.values())
            
            fig.add_trace(
                go.Bar(
                    x=brands,
                    y=similarities,
                    marker_color=similarities,
                    colorscale='Viridis'
                )
            )
            
            fig.update_layout(
                title="Average Similarity Scores by Brand",
                xaxis_title="Brand",
                yaxis_title="Average Similarity Score",
                width=1200,
                height=600,
                template='plotly_white'
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            # Save visualization
            output_path = os.path.join(
                self.viz_dirs['similarity'], 
                "similarity_summary.html"
            )
            fig.write_html(output_path)
            print("Saved similarity summary visualization")
            
        except Exception as e:
            print(f"Error creating similarity summary: {str(e)}")
def main():
    try:
        analyzer = WatchEmbeddingAnalyzer()
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()

