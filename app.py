import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_system_core_assets():
    try:
        with open("spotify_hybrid_bundle.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

system_assets = load_system_core_assets()

st.set_page_config(page_title="VibeStream UI", page_icon="🎵", layout="wide")
st.title("🎵 VibeStream: Next-Gen Music Recommendation Engine")

# File upload
st.sidebar.header("📁 Operational Pipeline Input")
user_csv_file = st.sidebar.file_uploader("Drop custom music catalog CSV (Optional override)", type=["csv"])

# Target feature list
acoustic_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

if user_csv_file is not None:
    st.sidebar.success("Custom CSV parsed successfully.")
    raw_custom_data = pd.read_csv(user_csv_file)
    spotify_df = raw_custom_data.dropna(subset=['track_id', 'track_name', 'track_artist'] if 'track_id' in raw_custom_data.columns else None)
    if 'track_id' in spotify_df.columns:
        spotify_df = spotify_df.drop_duplicates(subset=['track_id'])
    spotify_df = spotify_df.reset_index(drop=True)
    
    active_features = [m for m in acoustic_features if m in spotify_df.columns]
    if len(active_features) > 3:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(spotify_df[active_features])
        for i, feat in enumerate(active_features):
            spotify_df[f'norm_{feat}'] = normalized_data[:, i]
        custom_data_active = True
    else:
        st.error("Uploaded CSV file misses critical structural metrics.")
        spotify_df = system_assets['spotify_df']
        custom_data_active = False
else:
    if system_assets:
        spotify_df = system_assets['spotify_df']
        custom_data_active = False
    else:
        st.error("No baseline dataset found.")
        st.stop()

engine_selection = st.sidebar.selectbox("Active Pipeline Routine", ["Acoustic Content Similarity"] if custom_data_active else ["Acoustic Content Similarity", "Behavioral Neighborhood Matching"])

if "Acoustic Content Similarity" in engine_selection:
    st.subheader("Compute Track Relationships by Structural Signature Profiles")
    name_key = 'track_name' if 'track_name' in spotify_df.columns else spotify_df.columns[0]
    artist_key = 'track_artist' if 'track_artist' in spotify_df.columns else None
    
    unique_tracks_list = spotify_df[name_key].unique()
    selected_track = st.selectbox("Search or target an engine track:", unique_tracks_list)
    
    if st.button("Generate Recommendations"):
        target_idx = spotify_df[spotify_df[name_key] == selected_track].index[0]
        
        # LIVE COMPUTE: Pure matrix ki jagah sirf single row vs poor vector ka similarity live nikalna! (Super fast & 0 MB storage)
        norm_cols = [f'norm_{f}' for f in acoustic_features if f'norm_{f}' in spotify_df.columns]
        target_vector = spotify_df.loc[[target_idx], norm_cols].values
        all_vectors = spotify_df[norm_cols].values
        
        live_sim_scores = cosine_similarity(target_vector, all_vectors).flatten()
        match_scores = sorted(list(enumerate(live_sim_scores)), key=lambda x: x[1], reverse=True)
        
        st.write("### Target matches found matching vector criteria:")
        for ranked_item in match_scores[1:6]:
            row_data = spotify_df.iloc[ranked_item[0]]
            artist_display = f" by *{row_data[artist_key]}*" if artist_key else ""
            st.write(f"- **{row_data[name_key]}**{artist_display} `[Match: {ranked_item[1]:.2%}]`")

else:
    st.subheader("Track Identifiers Derived via Collaborative Signature Preferences")
    user_index_query = st.number_input("Target active User ID tracker map:", min_value=1001, max_value=1100, value=1001, step=1)
    
    if st.button("Run Collaborative Match Matrix"):
        pivot_data = system_assets['behavioral_pivot_matrix']
        sim_user_lookup = system_assets['behavioral_sim_df']
        
        if user_index_query in pivot_data.index:
            peer_profiles = sim_user_lookup[user_index_query].sort_values(ascending=False).index[1:6]
            mean_peer_predictions = pivot_data.loc[peer_profiles].mean(axis=0)
            unseen_tracks_mask = pivot_data.loc[user_index_query] == 0
            
            top_collaborative_recommendations = mean_peer_predictions[unseen_tracks_mask].sort_values(ascending=False).head(5)
            
            st.write(f"### Top Picks Generated:")
            for recommended_track_id in top_collaborative_recommendations.index:
                track_metadata = spotify_df[spotify_df['track_id'] == recommended_track_id]
                if not track_metadata.empty:
                    st.write(f"- **{track_metadata['track_name'].values[0]}** by *{track_metadata['track_artist'].values[0]}*")
        else:
            st.warning("Selected user index out of bounds.")