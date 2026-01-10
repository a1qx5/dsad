import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


songs = pd.read_csv("data/IN/cleaned_songs.csv")

features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms']

X = songs[features]

# Check correlations
corr_matrix = X.corr()
print(f"Correlation matrix (Pearson):\n {corr_matrix}")
corr_matrix.to_csv('data/OUT/correlation_matrix.csv')
print('\n[INFO] Saved correlation matrix to data/OUT/correlation_matrix.csv')


# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save and add metadata back
X_scaled_df = pd.DataFrame({
    'track_id': songs['track_id'].values,
    'track_name': songs['track_name'].values,
    'track_artist': songs['track_artist'].values,
    'playlist_genre': songs['playlist_genre'].values,
    'playlist_subgenre': songs['playlist_subgenre'].values
})

# Add standardized features
for i, feature in enumerate(features):
    X_scaled_df[feature] = X_scaled[:, i]

X_scaled_df.to_csv('data/IN/scaled_features.csv', index=False)
print('\n[INFO] Saved standardized features to data/IN/scaled_features.csv')

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png')
plt.show()

# Histogram
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(features):
    axes[i].hist(X[feature], bins=50, edgecolor='black')
    axes[i].set_title(feature)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('figures/feature_distributions.png')
plt.show()

# Bar chart for genre distribution
plt.figure(figsize=(10, 6))
songs['playlist_genre'].value_counts().plot(kind='bar')
plt.title('Distribution of Songs by Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/genre_distribution.png')
plt.show()

print("\n[INFO] Saved correlation heatmap, histigram and genre distribution bar chart to 'figures' folder")