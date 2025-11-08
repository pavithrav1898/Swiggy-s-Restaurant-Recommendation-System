import streamlit as st
import pandas as pd
import pickle

# Load cleaned data globally
cleaned = pd.read_csv("cleaned_data.csv")

# Load encoder, scaler, and model once globally for performance
@st.cache_resource
def load_resources():
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    return encoder, scaler, knn

encoder, scaler, knn = load_resources()

def recommend_restaurants(city, cuisine, rating, cost):
    # Use mean rating_count from cleaned as default
    mean_rating_count = cleaned["rating_count"].mean()

    # Prepare user input dataframes
    user_num_df = pd.DataFrame([[rating, mean_rating_count, cost]], columns=['rating', 'rating_count', 'cost'])
    user_cat_df = pd.DataFrame([[city, cuisine]], columns=['city', 'cuisine'])

    # Encode categorical features
    encoded_cats = encoder.transform(user_cat_df)
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(['city', 'cuisine']))

    # Scale numerical features
    scaled_num = scaler.transform(user_num_df)
    scaled_num_df = pd.DataFrame(scaled_num, columns=['rating', 'rating_count', 'cost'])

    # Combine scaled numerical and encoded categorical features
    user_final = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

    # Find nearest neighbors using KNN
    distances, indices = knn.kneighbors(user_final)

    # Exclude first neighbor as the input itself, get recommendations
    recommended_indices = indices[0][1:]
    recommendations = cleaned.iloc[recommended_indices]

    # Filter recommendations to exact city match
    recommendations = recommendations[recommendations['city'] == city]

    # Return subset with relevant columns
    return recommendations[['name', 'city', 'rating', 'rating_count', 'cost', 'cuisine']]

# Streamlit UI
st.title("Swiggy Restaurant Recommendation System")

city = st.selectbox("Select City", sorted(cleaned['city'].unique()))
cuisine = st.selectbox("Select Cuisine", sorted(cleaned['cuisine'].unique()))

rating = st.slider("Select Minimum Rating", 1.0, 5.0, 3.0, step=0.1)

cost_min = int(cleaned['cost'].min())
cost_max = int(cleaned['cost'].max())
cost = st.slider("Select Cost for Two Range", cost_min, cost_max, (cost_min, cost_max))

# When user clicks the button, get recommendations
if st.button("Get Recommendations"):
    # Try to find exact matches first in data
    exact_matches = cleaned[
        (cleaned['city'] == city) &
        (cleaned['cuisine'] == cuisine) &
        (cleaned['rating'] >= rating) &
        (cleaned['cost'] >= cost[0]) &
        (cleaned['cost'] <= cost[1])
    ]

    if not exact_matches.empty:
        st.subheader("Exact Matches Found")
        st.dataframe(exact_matches[['name', 'city', 'rating', 'rating_count', 'cost', 'cuisine']])
    else:
        st.info("No exact matches found. Showing nearest neighbor recommendations.")
        recs = recommend_restaurants(city, cuisine, rating, sum(cost) / 2)
        if recs.empty:
            st.error("No recommendations available for the given inputs.")
        else:
            st.subheader("Recommended Restaurants (Nearest Neighbors)")
            st.dataframe(recs)
