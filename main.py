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
from sklearn.metrics import mean_squared_error as MSE
from ast import literal_eval


#get director from the crew
def getDirector(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top n elements or entire list; whichever is more.
def getList(x):
    number_of_items = 1
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > number_of_items:
            names = names[:number_of_items]
        return names

    #Return empty list in case of missing/malformed data
    return []

def processData(nrows):
    # whole data is more that 46 000 rows, we need to cut it

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

    # merge data
    data = pd.merge(movies_metadata, credits, on='id', how='inner')
    data = pd.merge(data, keywords, on='id', how='inner')

    # budget cannot be 0 or not int
    data = data[pd.to_numeric(data['budget'], errors='coerce').notnull()]
    data['budget'] = data['budget'].astype(int)
    data = data[(data[['budget']] > 1000).all(axis=1)]

    # parse JSON into: top3 actors, director, related genres, keywords
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        data[feature] = data[feature].apply(literal_eval)
    data['director'] = data['crew'].apply(getDirector)
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        data[feature] = data[feature].apply(getList)

    # set directors as columns
    data['director'].replace(' ', '_', regex=True, inplace=True)
    data = pd.get_dummies(data, columns=['director'])

    # set language as columns
    data['original_language'].replace(' ', '_', regex=True, inplace=True)
    data = pd.get_dummies(data, columns=['original_language'])

    # set actors as columns
    tmp = data['cast']
    tmp = pd.get_dummies(tmp.apply(pd.Series).stack()).sum(level=0)
    data = pd.concat([data, tmp], axis=1)

    # set genres as columns
    tmp = data['genres']
    tmp = pd.get_dummies(tmp.apply(pd.Series).stack()).sum(level=0)
    data = pd.concat([data, tmp], axis=1)

    # set keywords as columns
    tmp = data['keywords']
    tmp = pd.get_dummies(tmp.apply(pd.Series).stack()).sum(level=0)
    data = pd.concat([data, tmp], axis=1)

    # drop unnecessary
    data.drop(['crew', 'id', 'title', 'release_date', 'cast', 'genres', 'keywords'],
              axis=1, inplace=True)

    data.to_csv('data/ready_data.csv')
    return

def XGB(size, depth, rounds, early_stoppnig):
    # data after initial processing
    data = pd.read_csv('data/ready_data.csv', nrows=size)
    data_Y = data['vote_average']
    data_X = data.drop('vote_average', axis=1, inplace=False)
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, train_size=0.3)


    # Instantiation
    xgb_r = xgb.XGBRegressor(objective='reg:squarederror', #reg:squarederror, reg:squaredlogerror, reg:squaredlogerror, reg:pseudohubererror
                             n_estimators=rounds,
                             #seed=123,
                             max_depth=depth,
                             booster='dart',#gbtree, gblinear, dart
                             )


    # Fitting the model
    xgb_r.fit(train_X, train_y, verbose=True,
              eval_set=[(test_X, test_y)],
              early_stopping_rounds=early_stoppnig,)

    # Predict the model
    pred = xgb_r.predict(test_X)

    # RMSE Computation
    rmse = np.sqrt(MSE(test_y, pred))
    print("RMSE : % f" % (rmse))

def NN():
    pass


if __name__ == '__main__':
    #processData(30000)

    XGB(2000, 20, 100, 20)




