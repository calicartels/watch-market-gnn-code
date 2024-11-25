# Watch Market GNN Code Repository

This repository contains the code implementation for creating and analyzing a Graph Neural Network (GNN) based dataset of the luxury watch market. The dataset itself is hosted on [Hugging Face](https://huggingface.co/datasets/TMVishnu/watch-market-gnn).

## Repository Structure

### `/code`
Contains the core implementation of the Graph Neural Network:
- `watch_gnn.py`: Primary GNN implementation including:
  - Data preprocessing
  - Feature engineering
  - Network construction
  - Embedding generation
  - Graph structure creation
 
  This is an overview of what functionality has been in the `watch_gnn.py` :

  ![High-Level Overview](https://raw.githubusercontent.com/calicartels/watch-market-gnn-code/main/images/1.png)


### `/visualization`
Contains analysis and visualization scripts:
- `watch_analysis.py`: Implements various visualizations including:
  - UMAP embeddings
  - t-SNE analysis
  - PCA visualization
  - Force-directed graph
  - Starburst visualization
  - Brand distribution analysis
  - Correlation studies

### `/images`
Contains visualization outputs:
1. Brand distribution treemap
2. Feature correlation matrix
3. UMAP visualization
4. t-SNE analysis
5. PCA visualization
6. Force-directed graph
7. Starburst graph
8. Network architecture diagram

### Root Files
- `requirements.txt`: Lists all Python dependencies
- `Watches.csv`: Original dataset file
- `.gitignore`: Specifies files and directories to be ignored by Git

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/calicartels/watch-market-gnn-code.git
cd watch-market-gnn-code
```
2. Create and activate a virtual environment:
```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage:

1. For GNN model implementation:

```bash
python code/watch_gnn.py
```

2. For visualization and analysis:

```bash
python visualization/watch_analysis.py
```

## Dataset Access:
The complete dataset is available on Hugging Face: [TMVishnu/watch-market-gnn](https://huggingface.co/datasets/TMVishnu/watch-market-gnn)

## Note:

This complete code took me close to 6 hours to run without a GPU on a Macbook Air M3 with 16GB RAM. 
Wait for the complete code to finish running.
