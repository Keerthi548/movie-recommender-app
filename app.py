import pandas as pd
from surprise import SVD
import gradio as gr
import pickle

# ---------- Load your model & data -------------
# Make sure 'svd_model.pkl' and 'movies.csv' are in the same folder as this app.py file.

# Load the trained SVD model
with open("svd_model.pkl", "rb") as f:
    algo = pickle.load(f)

# Load movies dataset
movies = pd.read_csv("movies.csv")

# ---------- Define Recommender Function ----------
def recommend_movies(user_id):
    """
    Generate top-10 movie recommendations for a given user_id.
    """
    recs = []
    # Predict ratings for all movies
    for movieId in movies['movieId'].tolist():
        pred = algo.predict(user_id, movieId)
        recs.append((movieId, pred.est))
    
    # Sort by predicted rating (descending order)
    top10 = sorted(recs, key=lambda x: x[1], reverse=True)[:10]

    # Convert to DataFrame for easy display
    top10_df = pd.DataFrame(top10, columns=["movieId", "pred_rating"])
    top10_df = pd.merge(top10_df, movies, on="movieId", how="left")

    # Return only relevant columns
    return top10_df[['title', 'pred_rating']]

# ---------- Create Gradio UI ----------
def show_recommendations(user_id):
    try:
        df = recommend_movies(int(user_id))
        return df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

app = gr.Interface(
    fn=show_recommendations,
    inputs=gr.Number(label="Enter User ID"),
    outputs="dataframe",
    title="🎬 Movie Recommendation System (SVD)",
    description="Enter a user ID to get Top-10 movie recommendations using SVD Collaborative Filtering."
)

# ---------- Run App ----------
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=10000)
