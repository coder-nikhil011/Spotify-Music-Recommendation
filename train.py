import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

print("--- Step 1: Loading Data ---")
source_csv_path = "spotify dataset.csv"
if not os.path.exists(source_csv_path):
    raise FileNotFoundError(f"Please put '{source_csv_path}' in this folder!")

raw_spotify_df = pd.read_csv(source_csv_path)
print(f"Loaded {raw_spotify_df.shape[0]} songs.")

print("\n--- Step 2: Cleaning Data ---")
sanitized_df = raw_spotify_df.dropna(subset=['track_id', 'track_name', 'track_artist'])
sanitized_df = sanitized_df.drop_duplicates(subset=['track_id']).reset_index(drop=True)

acoustic_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                     'speechiness', 'acousticness', 'instrumentalness', 
                     'liveness', 'valence', 'tempo']

min_max_transformer = MinMaxScaler()
# normalized vectors ko dataframe mein hi save kar lete hain
normalized_vectors = min_max_transformer.fit_transform(sanitized_df[acoustic_features])
for i, feature in enumerate(acoustic_features):
    sanitized_df[f'norm_{feature}'] = normalized_vectors[:, i]

print("\n--- Step 3: Simulating User Ratings ---")
np.random.seed(42)
target_pool_size = 100 # Size chota kiya taaki matrix memory optimize rahe
simulated_log_volume = 5000

mock_interaction_matrix = {
    'user_id': np.random.randint(1001, 1001 + target_pool_size, size=simulated_log_volume),
    'track_id': np.random.choice(sanitized_df['track_id'], size=simulated_log_volume),
    'affinity_score': np.random.randint(1, 6, size=simulated_log_volume)
}

user_affinity_df = pd.DataFrame(mock_interaction_matrix)
user_affinity_df = user_affinity_df.drop_duplicates(subset=['user_id', 'track_id']).reset_index(drop=True)

print("\n--- Step 4: Building Recommendation Models ---")
behavioral_pivot_matrix = user_affinity_df.pivot(index='user_id', columns='track_id', values='affinity_score').fillna(0)
behavioral_similarity = cosine_similarity(behavioral_pivot_matrix)
behavioral_sim_df = pd.DataFrame(behavioral_similarity, index=behavioral_pivot_matrix.index, columns=behavioral_pivot_matrix.index)

print("\n--- Step 5: Exporting Compact Model Artifacts ---")
# CRITICAL FIX: Pure matrix ki jagah sirf features save karenge, matrix app live banayega!
deployment_bundle = {
    'spotify_df': sanitized_df,
    'behavioral_pivot_matrix': behavioral_pivot_matrix.astype(np.float32),
    'behavioral_sim_df': behavioral_sim_df.astype(np.float32),
    'acoustic_features': acoustic_features
}

with open("spotify_hybrid_bundle.pkl", "wb") as storage_file:
    pickle.dump(deployment_bundle, storage_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Export Complete! 'spotify_hybrid_bundle.pkl' size is now ultra-lightweight!")