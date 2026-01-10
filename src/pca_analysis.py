import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# 1. LOAD DATA
print("[INFO] Loading standardized features...")
scaled_data = pd.read_csv("data/IN/scaled_features.csv")

# Separate metadata from features
metadata_cols = ['track_id', 'track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre']
feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'duration_ms']

metadata = scaled_data[metadata_cols]
X_scaled = scaled_data[feature_cols].values

print(f"Dataset: {X_scaled.shape[0]} songs with {X_scaled.shape[1]} features\n")


# 2. PERFORM PCA
print("[INFO] Applying PCA...")
pca = PCA()  # Calculate all components
X_pca = pca.fit_transform(X_scaled)

# Calculate variance explained
variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print("\nVariance Explained by Principal Components:")
print(f"{'Component':<12} {'Individual %':<15} {'Cumulative %':<15}")
print("-" * 42)
for i in range(len(variance_explained)):
    print(f"PC{i+1:<11} {variance_explained[i]*100:>8.2f}%      {cumulative_variance[i]*100:>8.2f}%")

# Determine optimal number of components (85% variance threshold)
n_components_optimal = np.argmax(cumulative_variance >= 0.85) + 1
print(f"\n[RESULT] Using {n_components_optimal} components explains {cumulative_variance[n_components_optimal-1]*100:.2f}% of variance\n")


# 3. ANALYZE COMPONENT LOADINGS
print("[INFO] Analyzing component loadings...")
# Loadings show how much each original feature contributes to each PC
loadings = pd.DataFrame(
    pca.components_[:4].T,  # First 4 components only for simplicity
    columns=['PC1', 'PC2', 'PC3', 'PC4'],
    index=feature_cols
)

print("\nFeature Loadings (First 4 Components):")
print(loadings.round(3))

# Save loadings
loadings_full = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(12)],
    index=feature_cols
)
loadings_full.to_csv('data/OUT/pca_loadings.csv')
print("\n[SAVED] data/OUT/pca_loadings.csv")


# 4. VISUALIZATION 1: Scree Plot (Variance Explained)
print("\n[INFO] Creating visualizations...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
ax1.bar(range(1, 13), variance_explained * 100, color='steelblue', alpha=0.7, edgecolor='black')
ax1.plot(range(1, 13), variance_explained * 100, 'ro-', linewidth=2)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Variance Explained (%)', fontsize=12)
ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, 13))
ax1.grid(axis='y', alpha=0.3)

# Cumulative variance
ax2.plot(range(1, 13), cumulative_variance * 100, 'go-', linewidth=2, markersize=8)
ax2.axhline(y=85, color='red', linestyle='--', linewidth=2, label='85% threshold')
ax2.fill_between(range(1, 13), cumulative_variance * 100, alpha=0.2, color='green')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Variance (%)', fontsize=12)
ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, 13))
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/pca_variance.png', dpi=300, bbox_inches='tight')
print("[SAVED] figures/pca_variance.png")
plt.show()


# 5. VISUALIZATION 2: Component Loadings Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(loadings.T, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, linewidths=1, cbar_kws={'label': 'Loading Value'})
plt.title('Feature Loadings on First 4 Principal Components', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
plt.tight_layout()
plt.savefig('figures/pca_loadings.png', dpi=300, bbox_inches='tight')
print("[SAVED] figures/pca_loadings.png")
plt.show()


# 6. VISUALIZATION 3: 2D Scatter Plot (PC1 vs PC2 by Genre)
plt.figure(figsize=(12, 8))

genres = metadata['playlist_genre'].unique()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

for genre, color in zip(genres, colors):
    mask = metadata['playlist_genre'] == genre
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=genre, alpha=0.5, s=15, color=color, edgecolors='black', linewidth=0.2)

plt.xlabel(f'PC1 ({variance_explained[0]*100:.2f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({variance_explained[1]*100:.2f}% variance)', fontsize=12)
plt.title('PCA Results: Songs Projected onto First Two Principal Components', fontsize=14, fontweight='bold')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.3)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pca_scatter.png', dpi=300, bbox_inches='tight')
print("[SAVED] figures/pca_scatter.png")
plt.show()


# 7. SAVE TRANSFORMED DATA FOR CLUSTERING
print("\n[INFO] Saving PCA-transformed data...")

# Save with optimal number of components (for clustering)
pca_df = pd.DataFrame(
    X_pca[:, :n_components_optimal],
    columns=[f'PC{i+1}' for i in range(n_components_optimal)]
)
pca_result = pd.concat([metadata.reset_index(drop=True), pca_df], axis=1)
pca_result.to_csv('data/OUT/pca_data.csv', index=False)
print(f"[SAVED] data/OUT/pca_data.csv ({n_components_optimal} components)")


# 8. SUMMARY
print("\n" + "=" * 70)
print("PCA ANALYSIS COMPLETE")
print("=" * 70)
print(f"Original features: {len(feature_cols)}")
print(f"Reduced to: {n_components_optimal} principal components")
print(f"Variance retained: {cumulative_variance[n_components_optimal-1]*100:.2f}%")
print(f"Total songs analyzed: {X_scaled.shape[0]:,}")
print("\nOutputs generated:")
print("  • figures/pca_variance.png      (variance explained)")
print("  • figures/pca_loadings.png      (feature contributions)")
print("  • figures/pca_scatter.png       (2D visualization by genre)")
print("  • data/OUT/pca_loadings.csv     (detailed loadings)")
print("  • data/OUT/pca_data.csv         (transformed data for clustering)")
print("=" * 70)
