import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns

# HELPER FUNCTIONS

def find_optimal_threshold(linkage_matrix):
    """
    Find the optimal cutting point by identifying where the largest
    jump in aggregation distance occurs.

    Returns: threshold value, junction index, total number of junctions
    """
    n = linkage_matrix.shape[0]  # number of merges/junctions

    # Distances at which merges occur (3rd column of linkage matrix)
    distances = linkage_matrix[:, 2]

    # Calculate differences between consecutive merge distances
    dist_later = distances[1:n]
    dist_earlier = distances[0:n - 1]
    differences = dist_later - dist_earlier

    # Find junction with maximum difference (biggest "jump")
    j = np.argmax(differences)

    # Threshold is midpoint between the two merge distances
    threshold = (distances[j] + distances[j + 1]) / 2

    return threshold, j, n


def get_cluster_labels(linkage_matrix, k):
    """
    Assign cluster labels based on cutting the dendrogram to get k clusters.
    Returns list of cluster labels as strings (e.g., 'C1', 'C2', ...)
    """
    cluster_ids = fcluster(linkage_matrix, k, criterion='maxclust')
    return ['C' + str(i) for i in cluster_ids]

# 1. LOAD DATA

print("=" * 70)
print("HIERARCHICAL CLUSTERING ANALYSIS")
print("=" * 70)

print("\nLoading PCA-transformed data...")
pca_data = pd.read_csv("data/OUT/pca_data.csv")

# Separate metadata from principal components
metadata_cols = ['track_id', 'track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre']
pc_cols = [col for col in pca_data.columns if col.startswith('PC')]

X = pca_data[pc_cols].values

print(f"Dataset: {X.shape[0]:,} songs with {X.shape[1]} principal components")
print(f"Principal components used: {pc_cols}")

# 2. PERFORM HIERARCHICAL CLUSTERING

print("\nPerforming hierarchical clustering (Ward's method)...")

# Compute linkage matrix using Ward's method
# Ward's method minimizes the total within-cluster variance
linkage_matrix = linkage(X, method='ward')

print("Linkage matrix computed successfully!")
print(f"       Shape: {linkage_matrix.shape}")
print(f"       (Each row represents a merge: [cluster1, cluster2, distance, n_observations])")

# 3. DETERMINE OPTIMAL NUMBER OF CLUSTERS

print("\nFinding optimal number of clusters...")

threshold, junction_idx, n_junctions = find_optimal_threshold(linkage_matrix)
optimal_k = n_junctions - junction_idx  # Number of clusters at optimal cut

print(f"       Total merges (junctions): {n_junctions}")
print(f"       Maximum distance jump at junction: {junction_idx}")
print(f"       Optimal threshold: {threshold:.4f}")
print(f"       Optimal number of clusters: {optimal_k}")

# Also show the top 5 largest jumps for reference
distances = linkage_matrix[:, 2]
differences = np.diff(distances)
top_5_jumps = np.argsort(differences)[-5:][::-1]

print("\n       Top 5 largest distance jumps:")
for i, j in enumerate(top_5_jumps):
    k_at_jump = n_junctions - j
    print(f"         {i + 1}. Junction {j}: jump = {differences[j]:.4f} -> {k_at_jump} clusters")

# 4. ASSIGN CLUSTER LABELS

print(f"\nAssigning songs to {optimal_k} clusters...")
cluster_labels = get_cluster_labels(linkage_matrix, optimal_k)
pca_data['cluster'] = cluster_labels

# Show cluster distribution
cluster_counts = pca_data['cluster'].value_counts().sort_index()
print("\nCluster distribution:")
for cluster, count in cluster_counts.items():
    print(f"       {cluster}: {count:,} songs ({count / len(pca_data) * 100:.1f}%)")

# 5. VISUALIZATION 1: TRUNCATED DENDROGRAM

print("\nCreating dendrogram visualization...")

plt.figure(figsize=(16, 10))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # Show only last p merged clusters
    p=50,  # Number of clusters to show at leaf level
    show_leaf_counts=True,  # Show count of observations in each leaf
    leaf_rotation=90,
    leaf_font_size=9,
    color_threshold=threshold,
    above_threshold_color='gray'
)
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
            label=f'Threshold = {threshold:.2f} ({optimal_k} clusters)')
plt.title(f'Hierarchical Clustering Dendrogram ({len(pca_data):,} songs)',
          fontsize=16, fontweight='bold')
plt.xlabel('Cluster (number of songs in parentheses)', fontsize=12)
plt.ylabel('Distance (Ward)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('figures/dendrogram.png', dpi=300, bbox_inches='tight')
print("figures/dendrogram.png")
plt.show()

# 6. VISUALIZATION 2: DISTANCE GROWTH / ELBOW PLOT

print("\nCreating elbow plot...")

# Show last 50 merges to visualize where the big jumps occur
last_n = 50
merge_distances = linkage_matrix[-last_n:, 2]
n_clusters_range = list(range(last_n, 0, -1))

plt.figure(figsize=(12, 6))
plt.plot(n_clusters_range, merge_distances, 'bo-', linewidth=2, markersize=6)
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
            label=f'Optimal threshold ({optimal_k} clusters)')
plt.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Optimal k = {optimal_k}')
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Merge Distance (Ward)', fontsize=12)
plt.title('Cluster Merge Distance vs Number of Clusters (Elbow Plot)', fontsize=14, fontweight='bold')
plt.gca().invert_xaxis()
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/cluster_elbow_plot.png', dpi=300, bbox_inches='tight')
print("figures/cluster_elbow_plot.png")
plt.show()

# 7. VISUALIZATION 3: CLUSTERS IN PCA SPACE (2D)

print("\n[Creating 2D cluster visualization...")

plt.figure(figsize=(14, 10))

# Get unique clusters and assign colors
unique_clusters = sorted(pca_data['cluster'].unique(), key=lambda x: int(x[1:]))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

for cluster, color in zip(unique_clusters, colors):
    mask = pca_data['cluster'] == cluster
    plt.scatter(
        X[mask, 0], X[mask, 1],
        label=f'{cluster} (n={mask.sum():,})',
        alpha=0.5, s=10, color=color
    )

plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title(f'Hierarchical Clustering Results ({len(pca_data):,} songs)',
          fontsize=14, fontweight='bold')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.3)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/clusters_pca_2d.png', dpi=300, bbox_inches='tight')
print("figures/clusters_pca_2d.png")
plt.show()

# 8. ANALYSIS: CLUSTERS VS GENRES

print("\nAnalyzing relationship between clusters and genres...")

# Create crosstab of clusters vs genres
cluster_genre_crosstab = pd.crosstab(
    pca_data['cluster'],
    pca_data['playlist_genre'],
    margins=True
)
print("\nCluster vs Genre Distribution:")
print(cluster_genre_crosstab)

# Save crosstab
cluster_genre_crosstab.to_csv('data/OUT/cluster_genre_crosstab.csv')
print("\n data/OUT/cluster_genre_crosstab.csv")

# Heatmap visualization
plt.figure(figsize=(12, 8))
crosstab_no_margins = pd.crosstab(pca_data['cluster'], pca_data['playlist_genre'])
crosstab_normalized = crosstab_no_margins.div(crosstab_no_margins.sum(axis=1), axis=0) * 100

sns.heatmap(
    crosstab_normalized,
    annot=True, fmt='.1f', cmap='YlOrRd',
    linewidths=0.5, cbar_kws={'label': 'Percentage within cluster (%)'}
)
plt.title('Genre Composition by Cluster (Row-Normalized %)', fontsize=14, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.tight_layout()
plt.savefig('figures/cluster_genre_heatmap.png', dpi=300, bbox_inches='tight')
print("figures/cluster_genre_heatmap.png")
plt.show()

# 9. SAVE RESULTS

print("\nSaving clustered data...")

pca_data.to_csv('data/OUT/clustered_songs.csv', index=False)
print("data/OUT/clustered_songs.csv")

# 10. SUMMARY

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING COMPLETE")
print("=" * 70)
print(f"Method: Ward's (minimum variance)")
print(f"Distance metric: Euclidean")
print(f"Input features: {len(pc_cols)} principal components")
print(f"Total songs clustered: {len(pca_data):,}")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Threshold (cutting distance): {threshold:.4f}")
print("\nCluster sizes:")
for cluster, count in cluster_counts.items():
    print(f"  - {cluster}: {count:,} songs ({count / len(pca_data) * 100:.1f}%)")
print("\nOutputs generated:")
print("  - figures/dendrogram.png             (dendrogram)")
print("  - figures/cluster_elbow_plot.png     (distance vs k)")
print("  - figures/clusters_pca_2d.png        (2D scatter by cluster)")
print("  - figures/cluster_genre_heatmap.png  (cluster-genre relationship)")
print("  - data/OUT/clustered_songs.csv       (data with cluster labels)")
print("  - data/OUT/cluster_genre_crosstab.csv (cluster vs genre table)")
print("=" * 70)