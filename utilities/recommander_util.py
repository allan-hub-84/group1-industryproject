from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import pandas as pd 


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

    #Remove the space into postal code 
    top_users_events["postalcode"] = top_users_events["postalcode"].astype(str).str.strip()

    # Get user location
    user_postal = str(data.get("postal_code", "")).strip()
    user_coords = postal_coords.get(user_postal)
    if not user_coords:
        return {"error": "User postal code not found in dataset."}, 400

    # Merge with user info name and email
    user_info = users_df[['customer_ID', 'first_name', 'last_name', 'email']]
    merged = top_users_events.merge(user_info, left_on='organizer_id', right_on='customer_ID', how='left')
    merged = merged.drop(columns=['customer_ID'])
    
    # Return results
    return merged.to_dict(orient="records"), 200