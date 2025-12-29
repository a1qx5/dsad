import pandas as pd

songs = pd.read_csv("data/raw/spotify_songs.csv")
print(f"Raw CSV shape : {songs.shape}\n")
print(f"Raw CSV columns:{songs.columns}\n")

# Removing irrelevant columns
irrelevant_columns = ['track_popularity', 'track_album_id', 'track_album_name', 'track_album_release_date', 'playlist_name', 'playlist_id']
print(f"[INFO] Removing irrelevant columns ({irrelevant_columns})\n")
cleaned_songs = songs.drop(columns=irrelevant_columns)

# Remove duplicated rows (first by id, then by name + artist)
duplicate_rows_by_id = cleaned_songs.duplicated(subset='track_id')
cleaned_songs = cleaned_songs.drop_duplicates(subset='track_id')
duplicate_rows_by_name = cleaned_songs.duplicated(subset=['track_artist', 'track_name'])
cleaned_songs = cleaned_songs.drop_duplicates(subset=['track_artist', 'track_name'])
print(f"[INFO] Removed {duplicate_rows_by_id.sum() + duplicate_rows_by_name.sum()} duplicate tracks\n")

# Handle missing values
missing_values = cleaned_songs.isnull().sum()
print("[INFO] Missing values per column:")
for col in cleaned_songs.columns:
    if missing_values[col] > 0:
        print(f"    {col}: {missing_values[col]}")

if missing_values.sum() == 0:
    print("[INFO] No missing values found\n")

# Only 1 row has missing values, drop it
rows_before = len(cleaned_songs)
cleaned_songs.dropna(inplace=True)
rows_dropped = rows_before - len(cleaned_songs)
print(f"\n[INFO] Dropped {rows_dropped} row(s) due to missing values\n")

# Assign correct data types to numeric columns
float_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
cleaned_songs[float_columns] = cleaned_songs[float_columns].astype(float)

int_columns = ['key', 'mode']
cleaned_songs[int_columns] = cleaned_songs[int_columns].astype(int)

print(f"Final CSV columns: {cleaned_songs.columns}\n")
print(f"Final CSV shape: {cleaned_songs.shape}\n")

# Save cleaned data
print("[INFO] Saving cleaned data to data/IN...")
cleaned_songs.to_csv("data/IN/cleaned_songs.csv", index=False)