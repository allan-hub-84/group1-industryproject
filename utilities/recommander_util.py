from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

def recommend_events(data, vectorizer, tfidf_matrix, users_df, events_df, postal_coords):
    # Combine the new user input questions
    new_text = " ".join(str(data.get(f"question_{i}", "")).replace(",", " ").strip() for i in range(1, 6))
    # Vectorize the new user based on the vectorizer we had used on our train data
    new_vector = vectorizer.transform([new_text])
    
    #cosine similarity between the new user and all existing users
    scores = cosine_similarity(new_vector, tfidf_matrix).flatten()
    
    sim_df = pd.DataFrame({
        'customer_ID': users_df['customer_ID'],  
        'similarity': scores 
    })
    
    top_users = sim_df.sort_values(by='similarity', ascending=False).head(10)
    top_user_ids = top_users['customer_ID'].tolist()

    top_users_events = events_df[events_df["organizer_id"].isin(top_user_ids)]

    return nearby_events.to_dict(orient="records"), 200
