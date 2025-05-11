from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from datetime import datetime
import pandas as pd
import numpy as np  

def is_within_radius(event_postal, user_coords, postal_coords, radius_km):
    """
    Check if an event's postal code is within km radius of the user's coordinates.
    """
    event_coords = postal_coords.get(str(event_postal).strip())
    if not event_coords:
        return False
    return geodesic(user_coords, event_coords).km <= radius_km

def get_user_vector(event_list, model):
    vectors = [model.wv[event] for event in event_list if event in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)



def recommend_events(data, vectorizer, tfidf_matrix, users_df, events_df):
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
    
    # Cosine Similarity threshold 0.7
    top_users = sim_df[sim_df['similarity'] > 0.7]

    # Cosine Similarity threshold 0.6 if the 0.7 threshold give us less than 10 users
    if len(top_users) < 10:
        top_users = sim_df[sim_df['similarity'] > 0.6]
    
    # Cosine Similarity threshold 0.5 if the 0.6 threshold give us less than 10 users
    if len(top_users) < 10:
        top_users = sim_df[sim_df['similarity'] > 0.5]
    
    top_user_ids = top_users['customer_ID'].tolist()

    print("threshold > 0.7:", len(sim_df[sim_df['similarity'] > 0.7]))
    print("threshold > 0.6:", len(sim_df[sim_df['similarity'] > 0.6]))
    print("Final selected users:", len(top_user_ids))

    # Remove the space from the poststal code
    top_users_events = events_df[events_df["organizer_id"].isin(top_user_ids)].copy()
    top_users_events["postalcode"] = top_users_events["postalcode"].str.replace(" ", "")
   
    # # Get current- user lat and lng
    # user_postal = str(data.get("postal_code", "")).strip()
    # user_coords = postal_coords.get(user_postal)
    # if not user_coords:
    #     return {"error": "User postal code not found in dataset."}, 400
    
    # Get the Events on and after the date
    # Filter events by date
    event_date = data.get("date")
    if event_date:
        event_date_converted = pd.to_datetime(event_date)
    else:
        event_date_converted = pd.to_datetime(datetime.now().date())

    top_users_events["converted_date"] = pd.to_datetime(top_users_events["date"])
    top_users_events = top_users_events[top_users_events["converted_date"] >= event_date_converted]


    # # Find the events near user defined radius
    # radius_km = float(data.get("radius", 10))  
    # top_users_events = top_users_events[
    #     top_users_events["postalcode"].apply(lambda p: is_within_radius(p, user_coords, postal_coords, radius_km))
    # ]

    # Merge with user info name and email
    user_info = users_df[['customer_ID', 'first_name', 'last_name', 'email', 'home_store_add','home_store_fsa']]
    top_users_events_new = top_users_events.merge(user_info, left_on='organizer_id', right_on='customer_ID', how='left')
    top_users_events_new = top_users_events_new.drop(columns=['customer_ID'])

    # Rename columns
    top_users_events_new = top_users_events_new.rename(columns={
        'first_name': 'organizer_first_name',
        'last_name': 'organizer_last_name',
        'email': 'organizer_email',
        'date': 'event_date',
        'home_store_add': 'organizer_home_store_add',
        'home_store_fsa' : 'organizer_home_store_postal',
        'address': 'event_address',
        'postalcode': 'event_postal_code',
        'time': 'event_time',
        'name': 'event_name',
        'loc_name': 'event_loc_name',
        'duration': 'event_duration'
    })
    
    # Return results
    return top_users_events_new.to_dict(orient="records"), 200