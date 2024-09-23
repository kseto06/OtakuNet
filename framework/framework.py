from typing import List, Dict
import numpy as np
import pandas as pd

# Function for zscore normalization of values
def zscore_normalization(df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # x_ij = (x_ij - mew_j) / stdev_j

    mean = np.zeros(len(genres))
    stdev = np.zeros(len(genres))

    # Calculate the mean
    for i, genre_col in enumerate(genres):
        mean[i] = np.sum(df[genre_col].values) / len(df[genre_col].values)

    # Calculate the standard deviation
    for i, genre_col in enumerate(genres):
        stdev[i] = np.sqrt(np.sum( (df[genre_col].values - mean[i])**2 ) / len(df[genre_col].values))
    
    # Normalize genres:
    for i, genre_col in enumerate(genres):
        df[genre_col] = (df[genre_col] - mean[i]) / stdev[i]

    return df

# Function for the MinMax Scaling of values
def MinMaxScaler(y: np.array, min = -1, max = 1):
    '''
    From sci-kit learn:

    The transformation is given by:

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

    '''
    y_std = (y - y.min()) / (y.max() - y.min())
    y_scaled = y_std * (max - min) + min
    return y_scaled

# Function for the Bayesian average 
def Bayesian_Rating(anime_df: pd.DataFrame):
    '''
    The Bayesian average can be computed as follows:

    Bayesian_rating = (mean_global_rating * C + sum of individ. ratings) / (C + number_of_ratings)
    C = constant representing 'weight' of the global mean
    '''

    # Compute C using the mean/avg num of ratings across all items:
    C = anime_df['Scored By'].mean()

    # Compute the global mean
    global_rating = anime_df['Score'].mean()

    # Sum of individual ratings = avg rating * num of ratings
    anime_df['sum_of_ratings'] = anime_df['Score'] * anime_df['Scored By']

    # Compute and append a new column to the anime df with bayesian_rating
    anime_df['Bayesian Rating'] = (
        (global_rating * C + anime_df['sum_of_ratings']) / (C + anime_df['Scored By'])
    )

    # Drop the sum_of_ratings here:
    anime_df.drop(columns=['sum_of_ratings'], inplace=True)
    
    return anime_df


# Function for the weighted average of the rating values
def weighted_average(df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # Calculate the weighted genre scores - this makes genres from highly-rated animes more significant
    for genre in genres:
        df[genre] = df[genre] * df['Bayesian Rating']

    # Aggregation and Normalization:
    # Now can sum user's genres / sum of ratings, and divide the score for normalization of weighted values
    user_genre_scores = df.groupby('user_id')[genres].sum()
    user_summed_ratings = df.groupby('user_id')['Bayesian Rating'].sum()
    user_genre_scores = user_genre_scores.div(user_summed_ratings, axis=0) #Divide the scores and ratings for the weighted average
    return user_genre_scores

# train_test_split function
def train_test_split(X, train_size, random_state = None, shuffle = True):
    '''
    Defines the split function to split training set, test set, or optional cv set
    Inspired by sci-kit learn's train_test_split function
    Input: 
    - X, y (data)
    - test_size (percentage of split)
    - random_state (reproducibility value)
    - shuffle (Optional shuffle of samples)
    '''

    #Set random seed for reproducibility -- ensures everytime code is run with the same data and params, same split acquired
    if random_state != None:
        np.random.seed(random_state)
    
    #get the number of training examples. (m,n) = (training example, features)
    m = X.shape[0] 

    #Calculate the number of test samples
    m_test = int(m * train_size)

    #Generate array of indices. Shuffling is optional, won't affect reproducibility with fixed random_state
    indices = np.arange(0, m, 1) #start, stop, step
    if shuffle: 
        np.random.shuffle(indices)
    
    #Split indices for train and test values
    test_indices = indices[:m_test] #values up to the number of wanted test values
    train_indices = indices[m_test:]

    #Return data in the form of X_train, X_test, y_train, y_test
    return X[train_indices], X[test_indices]