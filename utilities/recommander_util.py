from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import pandas as pd 

def is_within_radius(event_postal, user_coords, postal_coords, radius_km):
    """
    Check if an event's postal code is within km radius of the user's coordinates.
    """
    event_coords = postal_coords.get(str(event_postal).strip())
    if not event_coords:
        return False
    return geodesic(user_coords, event_coords).km <= radius_km



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
    
    top_users = sim_df.sort_values(by='similarity', ascending=False).head(20)
    top_user_ids = top_users['customer_ID'].tolist()

    # Remove the space from the poststal code
    top_users_events = events_df[events_df["organizer_id"].isin(top_user_ids)].copy()
    top_users_events["postalcode"] = top_users_events["postalcode"].str.replace(" ", "")
   
    # Get current- user lat and lng
    user_postal = str(data.get("postal_code", "")).strip()
    user_coords = postal_coords.get(user_postal)
    if not user_coords:
        return {"error": "User postal code not found in dataset."}, 400
    
    # Get the Events on and after the date
    event_date = data.get("date")
    if event_date:
        top_users_events["date"] = pd.to_datetime(top_users_events["date"])
        event_date_converted = pd.to_datetime(event_date)
        top_users_events = top_users_events[top_users_events["date"] >= event_date_converted]
    print(top_users_events.postalcode)


    # Find the events near user defined radius
    radius_km = float(data.get("radius", 10))  
    top_users_events = top_users_events[
        top_users_events["postalcode"].apply(lambda p: is_within_radius(p, user_coords, postal_coords, radius_km))
    ]

    # Merge with user info name and email
    user_info = users_df[['customer_ID', 'first_name', 'last_name', 'email']]
    top_users_events_new = top_users_events.merge(user_info, left_on='organizer_id', right_on='customer_ID', how='left')
    top_users_events_new = top_users_events_new.drop(columns=['customer_ID'])

    # Rename columns
    top_users_events_new = top_users_events_new.rename(columns={
        'first_name': 'organizer_first_name',
        'last_name': 'organizer_last_name',
        'email': 'organizer_email',
        'date': 'event_date'
    })
    
    # Return results
    return top_users_events_new.to_dict(orient="records"), 200