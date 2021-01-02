# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # in order to not show warning
import tensorflow as tf

from sklearn.model_selection import train_test_split
from ast import literal_eval


#get director from the crew
def getDirector(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def getList(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

def getAndProcessData():
    # whole data is more that 46 000 rows, we need to cut it
    nrows = 10000

    # read data
    credits = pd.read_csv('data/credits.csv', nrows=nrows)
    keywords = pd.read_csv('data/keywords.csv', nrows=nrows)
    movies_metadata = pd.read_csv('data/movies_metadata.csv',
                                  nrows=nrows,
                                  usecols=['id', 'genres', 'title', 'release_date', 'runtime', 'vote_average',
                                           'vote_count', 'budget', 'runtime', 'adult', 'original_language',
                                           'vote_average'])

    # just ints are correct ID
    movies_metadata = movies_metadata[pd.to_numeric(movies_metadata['id'], errors='coerce').notnull()]
    movies_metadata['id'] = movies_metadata['id'].astype(int)


    # merge data into one table so we can learn
    data = pd.merge(movies_metadata, credits, on='id', how='inner')
    data = pd.merge(data, keywords, on='id', how='inner')



    #parse JSON into: top3 actors, director, related genres, keywords
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        data[feature] = data[feature].apply(literal_eval)
    data['director'] = data['crew'].apply(getDirector)
    data.drop(['crew', 'id'], axis=1, inplace=True)#no longer needed
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        data[feature] = data[feature].apply(getList)


    # split
    data_Y = data['vote_average']
    data_X = data.drop('vote_average', axis=1, inplace=False)

    #print(data.shape)
    #print(data.head())
    #data.to_csv('tmp.csv')
    #data_X.to_csv('tmp2.csv')
    #data_Y.to_csv('tmp3.csv')





    return train_test_split(data_X, data_Y, train_size=0.3)

def XGB():

    pass

def NN():
    pass


if __name__ == '__main__':

    #data after initial processing
    #each algorithm can work on these and a process it further
    X_train, X_test, Y_train, Y_test = getAndProcessData()



