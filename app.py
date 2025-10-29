import pandas as pd
import gradio as gr

# ---------- Load dataset ----------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# ---------- Simple Recommender ----------
# Compute average rating per movie
movie_avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movie_avg_ratings = pd.merge(movie_avg_ratings, movies, on="movieId", how="left")
movie_avg_ratings = movie_avg_ratings.sort_values("rating", ascending=False)

# ---------- Define function ----------
def recommend_movies(user_id):
    """
    Return top 10 highest-rated movies (simple recommender).
    """
    top10_df = movie_avg_ratings.head(10)
    return top10_df[['title', 'rating']]

# ---------- Create Gradio UI ----------
app = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Number(label="Enter User ID"),
    outputs="dataframe",
    title="🎬 Movie Recommendation System",
    description="Simple Top-10 Movie Recommender based on average ratings (no Surprise dependency)."
)

# ---------- Run App ----------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(server_name="0.0.0.0", server_port=port)

