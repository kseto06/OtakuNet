from typing import List, Dict
import numpy as np
import pandas as pd

# Function for zscore normalization of values
def zscore_normalization(df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # x_ij = (x_ij - mew_j) / stdev_j
    # Calculate the mean
    mean = np.zeros(len(genres))
    stdev = np.zeros(len(genres))

    for i, genre_col in enumerate(genres):
        mean[i] = np.sum(df[genre_col].values) / len(df[genre_col].values)
    
    # Calculate the standard deviation
    for i, genre_col in enumerate(genres):
        stdev[i] = np.sqrt(np.sum( (df[genre_col].values - mean[i])**2 ) / len(df[genre_col].values))
    
    # Normalize genres:
    for i, genre_col in enumerate(genres):
        df[genre_col] = (df[genre_col] - mean[i]) / stdev[i]
    
    return df

# Function for the weighted average of the rating values
def weighted_average(df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # Calculate the weighted genre scores - this makes genres from highly-rated animes more significant
    for genre in genres:
        df[genre] = df[genre] * df['rating']

    # Aggregation and Normalization:
    # Now can sum user's genres / sum of ratings, and divide the score for normalization of weighted values
    user_genre_scores = df.groupby('user_id')[genres].sum()
    user_summed_ratings = df.groupby('user_id')['rating'].sum()
    user_genre_scores = user_genre_scores.div(user_summed_ratings, axis=0) #Divide the scores and ratings for the weighted average
    return user_genre_scores