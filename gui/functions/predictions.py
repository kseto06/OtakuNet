import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append('../framework')
from framework import *

from model import predict
from framework import MinMaxInverse

# Function to sort the prediction:
def prediction(genre_values: list):
    user_vector = np.array([genre_values]) #Create the user_vector for predicting
    print(user_vector.shape)
    genre_names = ['Action', 'Award Winning', 'Sci-Fi', 'Adventure', 'Drama',
       'Mystery', 'Supernatural', 'Fantasy', 'Sports', 'Comedy',
       'Romance', 'Slice of Life', 'Suspense', 'Gourmet',
       'Avant Garde', 'Horror', 'Girls Love', 'Boys Love']
    
    # Store the list of preferred genres
    preferred_genres = []
    for i in range(len(user_vector.flatten())):
        genre_rating = user_vector.flatten()[i]
        if genre_rating > 0:
            preferred_genres.append(genre_names[i])

    # Load item_prepared df from pd, pickle
    item_prepared = pd.DataFrame()
    with open('../notebooks/pickle/item_prepared.pkl', 'rb') as file:
        item_prepared = pickle.load(file)

    # User constants (mean, std)
    user_mean, user_std = 0., 0.
    with open('../notebooks/pickle/user_constants.pkl', 'rb') as file:
        user_mean, user_std = pickle.load(file)

    # User and item params retrieved from training
    user_params, item_params = dict(), dict()

    with open('../notebooks/pickle/user_model.pkl', 'rb') as file:
        user_params = pickle.load(file)

    with open('../notebooks/pickle/item_model.pkl', 'rb') as file:
        item_params = pickle.load(file)

    # y-original: Original ratings
    with open('../notebooks/pickle/y_original.pkl', 'rb') as file:
        y_original = pickle.load(file)

    # Item original -- original loaded item df
    with open('../notebooks/pickle/item_original.pkl', 'rb') as file:
        item_original = pickle.load(file)

    # Match the user vector to the number of movies in the dataset
    user_vecs = np.tile(user_vector.T, (item_prepared.to_numpy().shape[0])) #Transpose to prevent repetitions on original user vector
    user_vecs = user_vecs.T #Transpose back

    # Scale the user vector. Note that the item vector should already be scaled
    for genre in range(user_vecs.shape[1]):
        user_vecs[:, genre] = ((user_vecs[:, genre] - user_mean[genre]) / (user_std[genre]))

    # Get the item vector
    item_vecs = item_prepared
    item_vecs = item_vecs.drop(columns=['Bayesian Rating'])
    
    # Convert item_vecs to np for prediction
    item_vecs = item_vecs.to_numpy()

    print("Original item_vecs shape:", item_vecs.shape)
    print("After slicing:", item_vecs[:, 3:].shape)
    print("After transposing (if applicable):", item_vecs[:, 3:].T.shape)
    
    # Make a prediction
    new_pred = predict(item_vecs[:, 3:], user_vecs, user_params, item_params)

    # Inverse/Reverse Scaling
    new_pred_unscaled = MinMaxInverse(new_pred, min(y_original), max(y_original))

    # Sorting predictions
    sorted_i = np.argsort(-new_pred_unscaled, axis=0).reshape(-1).tolist() # Negate for descending order
    sorted_pred = new_pred_unscaled[sorted_i]
    sorted_items = item_vecs[sorted_i]

    # Generate sorted df
    sorted_items_df = pd.DataFrame(sorted_items, columns=['anime_id', 'Score', 'Scored By',	'Action', 'Award Winning', 'Sci-Fi', 'Adventure', 'Drama', 'Mystery', 'Supernatural', 'Fantasy', 'Sports', 'Comedy', 'Romance', 'Slice of Life', 'Suspense', 'Gourmet', 'Avant Garde', 'Horror', 'Girls Love', 'Boys Love'])
    sorted_items_df['anime_id'] = sorted_items_df['anime_id'].astype(int) #Convert anime_id to int
    top_items = sorted_items_df.merge(item_original[['anime_id', 'Name', 'Genres', 'Image URL']], on='anime_id', how='left')

    # Get rid of the genres in top_items
    top_items = top_items.drop(top_items.columns[1:21], axis=1)

    # Table generation
    pred_headers = [['anime_id', 'Anime Title', 'Genres', 'Image URL']]

    for i in range(0, 10): #Get the top 10 recommendations
        # Check if the predicted anime actually contains a genre that user has rated (likes)
        anime_genres = top_items.iloc[i]['Genres']
        genre_list = [genre.strip() for genre in anime_genres.split(',')]

        # If at least one genre the user likes is in the genre_list, we assume it's good to display on the table:

        # Iterate over preferred genres. If the genre in preferred genres is also in the genre_list, any(...) will return True. 
        if any(genre in genre_list for genre in preferred_genres): 
            # Round the predictions & actual ratings
            pred_headers.append([
                                sorted_items[i, 0].astype(int),
                                top_items.iloc[i]['Name'],
                                top_items.iloc[i]['Genres'],
                                top_items.iloc[i]['Image URL']
                                ])
    return pred_headers #Return predictions