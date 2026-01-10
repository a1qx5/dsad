import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# LOAD STANDARDIZED DATA
print("[INFO] Loading standardized features...")
scaled_data = pd.read_csv("data/IN/scaled_features.csv")

# Separate metadata from features
metadata_cols = ['track_id', 'track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre']
feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'duration_ms']

metadata = scaled_data[metadata_cols]
X_scaled = scaled_data[feature_cols].values

print(f"Dataset shape: {X_scaled.shape}")
print(f"Features: {feature_cols}\n")


# PERFORM PCA WITH ALL COMPONENTS
print("[INFO] Performing PCA with all 12 components...")
pca = PCA(n_components=12)
X_pca = pca.fit_transform(X_scaled)

# Variance explained by each component
variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print("\nVariance explained by each Principal Component:")
for i, var in enumerate(variance_explained, 1):
    print(f"  PC{i}: {var*100:.2f}% (Cumulative: {cumulative_variance[i-1]*100:.2f}%)")


# DETERMINE OPTIMAL NUMBER OF COMPONENTS
# Find number of components for 80%, 85%, 90%, 95% variance
thresholds = [0.80, 0.85, 0.90, 0.95]
print("\nComponents needed for variance thresholds:")
for threshold in thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  {threshold*100:.0f}% variance: {n_components} components")


# SCREE PLOT - Variance Explained
print("\n[INFO] Generating Scree Plot...")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
ax[0].bar(range(1, 13), variance_explained * 100, alpha=0.7, color='steelblue', edgecolor='black')
ax[0].plot(range(1, 13), variance_explained * 100, 'ro-', linewidth=2, markersize=6)
ax[0].set_xlabel('Principal Component', fontsize=12)
ax[0].set_ylabel('Variance Explained (%)', fontsize=12)
ax[0].set_title('Scree Plot - Variance per Component', fontsize=14, fontweight='bold')
ax[0].set_xticks(range(1, 13))
ax[0].grid(axis='y', alpha=0.3)

# Cumulative variance
ax[1].plot(range(1, 13), cumulative_variance * 100, 'go-', linewidth=2, markersize=6)
ax[1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
ax[1].axhline(y=90, color='orange', linestyle='--', label='90% threshold')
ax[1].set_xlabel('Number of Components', fontsize=12)
ax[1].set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
ax[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax[1].set_xticks(range(1, 13))
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/pca_scree_plot.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: figures/pca_scree_plot.png")
plt.show()


# COMPONENT LOADINGS ANALYSIS
print("\n[INFO] Analyzing component loadings...")
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'PC{i+1}' for i in range(12)],
    index=feature_cols
)

# Save loadings to CSV
loadings_df.to_csv('data/OUT/pca_loadings.csv')
print("[INFO] Saved: data/OUT/pca_loadings.csv")

# Print loadings for first 4 components
print("\nFeature Loadings for First 4 Principal Components:")
print(loadings_df.iloc[:, :4].to_string())


# LOADINGS HEATMAP
print("\n[INFO] Generating loadings heatmap...")
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df.T, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, linewidths=0.5, cbar_kws={'label': 'Loading'})
plt.title('PCA Feature Loadings Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
plt.tight_layout()
plt.savefig('figures/pca_loadings_heatmap.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: figures/pca_loadings_heatmap.png")
plt.show()


# 2D VISUALIZATION - First Two Principal Components
print("\n[INFO] Creating 2D scatter plot (PC1 vs PC2)...")
plt.figure(figsize=(12, 8))

# Color by genre
genres = metadata['playlist_genre'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))

for genre, color in zip(genres, colors):
    mask = metadata['playlist_genre'] == genre
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=genre, alpha=0.6, s=20, color=color, edgecolors='black', linewidth=0.3)

plt.xlabel(f'PC1 ({variance_explained[0]*100:.2f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({variance_explained[1]*100:.2f}% variance)', fontsize=12)
plt.title('PCA - First Two Principal Components by Genre', fontsize=14, fontweight='bold')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pca_2d_scatter.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: figures/pca_2d_scatter.png")
plt.show()


# 3D VISUALIZATION - First Three Principal Components
print("\n[INFO] Creating 3D scatter plot (PC1, PC2, PC3)...")
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for genre, color in zip(genres, colors):
    mask = metadata['playlist_genre'] == genre
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
               label=genre, alpha=0.6, s=20, color=color, edgecolors='black', linewidth=0.3)

ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.2f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.2f}%)', fontsize=11)
ax.set_zlabel(f'PC3 ({variance_explained[2]*100:.2f}%)', fontsize=11)
ax.set_title('PCA - First Three Principal Components by Genre', fontsize=14, fontweight='bold')
ax.legend(title='Genre', bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/pca_3d_scatter.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: figures/pca_3d_scatter.png")
plt.show()


# BIPLOT - PC1 vs PC2 with Feature Vectors
print("\n[INFO] Creating biplot...")
fig, ax = plt.subplots(figsize=(14, 10))

# Sample data for cleaner visualization (plot every 10th point)
sample_indices = np.random.choice(len(X_pca), size=2000, replace=False)
for genre, color in zip(genres, colors):
    mask = metadata['playlist_genre'] == genre
    sample_mask = np.array([i in sample_indices and mask.iloc[i] for i in range(len(mask))])
    ax.scatter(X_pca[sample_mask, 0], X_pca[sample_mask, 1], 
               label=genre, alpha=0.4, s=15, color=color)

# Plot feature vectors
scale_factor = 4  # Scale arrows for visibility
for i, feature in enumerate(feature_cols):
    ax.arrow(0, 0, 
             loadings[i, 0] * scale_factor, 
             loadings[i, 1] * scale_factor,
             head_width=0.15, head_length=0.15, fc='red', ec='red', linewidth=2, alpha=0.8)
    ax.text(loadings[i, 0] * scale_factor * 1.15, 
            loadings[i, 1] * scale_factor * 1.15,
            feature, fontsize=10, fontweight='bold', 
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.2f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.2f}% variance)', fontsize=12)
ax.set_title('PCA Biplot - Data Points and Feature Loadings', fontsize=14, fontweight='bold')
ax.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/pca_biplot.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: figures/pca_biplot.png")
plt.show()


# SAVE PCA-TRANSFORMED DATA FOR CLUSTERING
print("\n[INFO] Saving PCA-transformed data for clustering analysis...")

# Save data with optimal number of components (e.g., for 85% variance)
n_components_optimal = np.argmax(cumulative_variance >= 0.85) + 1
print(f"Using {n_components_optimal} components (explains {cumulative_variance[n_components_optimal-1]*100:.2f}% variance)")

# Create DataFrame with PCA results
pca_df = pd.DataFrame(
    X_pca[:, :n_components_optimal],
    columns=[f'PC{i+1}' for i in range(n_components_optimal)]
)

# Add metadata back
pca_result = pd.concat([metadata.reset_index(drop=True), pca_df], axis=1)
pca_result.to_csv('data/OUT/pca_transformed_data.csv', index=False)
print(f"[INFO] Saved: data/OUT/pca_transformed_data.csv ({n_components_optimal} components)")

# Also save full 12-component version
pca_full_df = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(12)]
)
pca_full_result = pd.concat([metadata.reset_index(drop=True), pca_full_df], axis=1)
pca_full_result.to_csv('data/OUT/pca_full_12components.csv', index=False)
print("[INFO] Saved: data/OUT/pca_full_12components.csv (all 12 components)")


# SUMMARY REPORT
print("\n" + "="*70)
print("PCA ANALYSIS SUMMARY")
print("="*70)
print(f"Original dimensions: {X_scaled.shape[1]} features")
print(f"Total samples: {X_scaled.shape[0]:,} songs")
print(f"\nRecommended components ({n_components_optimal}): {cumulative_variance[n_components_optimal-1]*100:.2f}% variance")
print(f"\nTop 3 components explain: {cumulative_variance[2]*100:.2f}% variance")
print(f"\n{'Component':<12} {'Variance %':<15} {'Cumulative %':<15}")
print("-"*42)
for i in range(min(6, len(variance_explained))):
    print(f"PC{i+1:<10} {variance_explained[i]*100:>8.2f}%      {cumulative_variance[i]*100:>8.2f}%")
print("="*70)

print("\n[SUCCESS] PCA analysis complete!")
print("\nGenerated outputs:")
print("  - figures/pca_scree_plot.png")
print("  - figures/pca_loadings_heatmap.png")
print("  - figures/pca_2d_scatter.png")
print("  - figures/pca_3d_scatter.png")
print("  - figures/pca_biplot.png")
print("  - data/OUT/pca_loadings.csv")
print("  - data/OUT/pca_transformed_data.csv")
print("  - data/OUT/pca_full_12components.csv")
