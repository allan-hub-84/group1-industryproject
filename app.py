from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from scipy import sparse

from utilities.recommander_util import recommend_events

# Load models and data
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = sparse.load_npz("models/tfidf_matrix.npz")
users_df = pd.read_pickle("models/users_df.pkl")
events_df = pd.read_csv("data/events.csv")

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    result, status = recommend_events(data, vectorizer, tfidf_matrix, users_df, events_df)
    return jsonify(result), status

if __name__ == "__main__":
    app.run(port=5000, debug=True)
