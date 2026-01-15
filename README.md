# Spotify Songs Analysis - PCA & Clustering

Analysis of 26,229 Spotify songs using Principal Component Analysis and Hierarchical Clustering.

## Dataset
**Source**: [Kaggle - Spotify Songs](https://www.kaggle.com/datasets/sujaykapadnis/spotify-songs/data)  
**Features**: 12 audio characteristics (danceability, energy, acousticness, etc.)  
**Genres**: 6 categories (pop, rap, rock, latin, EDM, R&B)

## Project Workflow

### 1. Data Cleaning & Normalization ✅
**Scripts**: `data_cleaning.py`, `data_corr_norm.py`
- Removed duplicates and irrelevant columns
- Handled missing values
- Standardized features (mean=0, std=1)
- **Output**: `data/IN/scaled_features.csv`

### 2. Principal Component Analysis (PCA) ✅
**Script**: `pca_analysis.py`
- Reduced 12 features to 9 principal components
- Retained 88.59% of variance
- Analyzed component loadings and interpretations
- **Outputs**: 
  - `data/OUT/pca_data.csv` (for clustering)
  - 3 visualizations in `figures/`

### 3. Hierarchical Clustering ✅
**Status**: Completed 

- Uses `data/OUT/pca_data.csv` as input
- Method: Ward's (minimum variance)
- Distance metric: Euclidean
- Input features: 9 principal components
- Assigned data to two clusters
- C1: 4 051 songs (15.4%)
- C2: 22 178 songs (84.6%)
- **Outputs**
  - `data/OUT/clustered_songs.csv` 
  - `data/OUT/cluster_genre_crosstab.csv`
  - 4 visualizations in `figures/`

### 4. Interpretation & Documentation ✅
**Status**: Complete

## How to Run

```bash
# Step 1: Data preparation
python src/data_cleaning.py
python src/data_corr_norm.py

# Step 2: PCA analysis
python src/pca_analysis.py

# Step 3: Hierarchical clustering
python src/hierarchical_clustering.py
```