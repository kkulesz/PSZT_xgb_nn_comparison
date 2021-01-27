import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import keras
from tensorflow.python.keras.applications.densenet import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # in order to not show warning
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from ast import literal_eval


# get director from the crew
def getDirector(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return 0


# Returns the list top n elements or entire list; whichever is more.
def getList(x):
    number_of_items = 3
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > number_of_items:
            names = names[:number_of_items]
        return names

    # Return empty list in case of missing/malformed data
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

    data.columns = data.columns.str.replace(' ', '_')
    # print(data.dtypes)
    data.to_csv('data/ready_data_NN.csv')
    return


def XGB(size):
    # data after initial processing
    data = pd.read_csv('data/ready_data.csv', nrows=size)

    # drop columns that all has the same value, because the do not give any information
    nunique = data.apply(pd.Series.nunique)
    columns_to_drop = nunique[nunique == 1].index
    data.drop(columns_to_drop, axis=1)

    # prepare data for regression
    data_Y = data['vote_average']
    data_X = data.drop('vote_average', axis=1, inplace=False)
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, train_size=0.3)
    # TODO:DMatrix

    params1 = {
        'max_depth': [10, 20, 30],
        'learning_rate': [0, 0.25, 1.0],
        'gamma': [0, 0.25, 1.0],
        'reg_lambda': [0, 1.0, 10.0],
        'scale_pos_weight': [1, 3, 5]
    }  # {'gamma': 1.0, 'learning_rate': 0.25, 'max_depth': 10, 'reg_lambda': 1.0, 'scale_pos_weight': 1}
    params2 = {
        'max_depth': [8, 10, 13],
        'learning_rate': [0.25],
        'gamma': [1.0, 2.0, 3.0],
        'reg_lambda': [1.0],
        'scale_pos_weight': [0.33, 0.66, 1.0]
    }  # {'gamma': 1.0, 'learning_rate': 0.25, 'max_depth': 13, 'reg_lambda': 1.0, 'scale_pos_weight': 0.33}

    params3 = {
        'max_depth': [12, 13, 14],
        'learning_rate': [0.25],
        'gamma': [1.0, 1.2, 1.4],
        'reg_lambda': [1.0],
        'scale_pos_weight': [0.25, 0.3, 0.35]
    }  # {'gamma': 1.2, 'learning_rate': 0.25, 'max_depth': 12, 'reg_lambda': 1.0, 'scale_pos_weight': 0.25}
    params4 = {
        'max_depth': [11, 12],
        'learning_rate': [0.25],
        'gamma': [1.2],
        'reg_lambda': [1.0],
        'scale_pos_weight': [0.22, 0.25, 0.28]
    }  # {'gamma': 1.2, 'learning_rate': 0.25, 'max_depth': 11, 'reg_lambda': 1.0, 'scale_pos_weight': 0.22}

    params5 = {
        'max_depth': [11],
        'learning_rate': [0.25],
        'gamma': [1.2],
        'reg_lambda': [1.0],
        'scale_pos_weight': [0.21, 0.22, 0.23]
    }  # {'gamma': 1.2, 'learning_rate': 0.25, 'max_depth': 11, 'reg_lambda': 1.0, 'scale_pos_weight': 0.21}

    params6 = {
        'max_depth': [11],
        'learning_rate': [0.25],
        'gamma': [1.2],
        'reg_lambda': [1.0],
        'scale_pos_weight': [0.15, 0.20]
    }

    params7 = {
        'max_depth': [11],
        'learning_rate': [0.25],
        'gamma': [1.2],
        'reg_lambda': [1.0],
        'scale_pos_weight': [0]
    }

    '''
    FOR HYPERPARAMETERS

    # Instantiation
    xgb_r = xgb.XGBRegressor(objective='reg:squarederror', #reg:squarederror, reg:squaredlogerror, reg:squaredlogerror, reg:pseudohubererror
                             #n_estimators=rounds,
                             missing=None,
                             booster='dart',#gbtree, gblinear, dart
                             )

    optimal = GridSearchCV(
        estimator=xgb_r,
        param_grid=params7,
        verbose=0,
        n_jobs=1,
        cv=5
    )
    optimal.fit(train_X, train_y, verbose=True,
                eval_set=[(test_X, test_y)],
                early_stopping_rounds=10,
                # eval_metric='aucpr'
                )
    print(optimal.best_params_)
    '''

    # training score
    # train_score = xgb_r.score(train_X, train_y)
    # print("Training score: ", train_score)

    # cross validation score
    # cross_validation_score = cross_val_score(xgb_r, train_X, train_y, cv=10)
    # print("Cross validation score: ", cross_validation_score)

    # Predict the model
    # pred = xgb_r.predict(test_X)

    # RMSE Computation
    # rmse = np.sqrt(MSE(test_y, pred))
    # print("RMSE : % f" % (rmse))


def NN(size):
    # data after initial processing
    data = pd.read_csv('data/ready_data_NN.csv', nrows=size)
    # drop columns that all has the same value, because the do not give any information
    nunique = data.apply(pd.Series.nunique)
    columns_to_drop = nunique[nunique == 1].index
    data.drop(columns_to_drop, axis=1)
    data.fillna(0, inplace=True)
    # print(data.isnull().any(axis=0))
    # return
    data_Y = data['vote_average']
    data_X = data.drop('vote_average', axis=1, inplace=False)
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, train_size=0.3)

    x_train = np.asarray(x_train).astype(np.float64)
    y_train = np.asarray(y_train).astype(np.float64)
    x_test = np.asarray(x_test).astype(np.float64)
    y_test = np.asarray(y_test).astype(np.float64)
    # create model
    inputs = keras.Input(shape=(x_train.shape[1],), name="digits")
    x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.Dense(32, activation="relu", name="dense_2")(x1)
    x2 = layers.LayerNormalization()(x2)
    outputs = layers.Dense(1, activation="relu", name="predictions")(x2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.mean_absolute_error])
    print(model.summary())

    print(x_train.shape)
    print(y_train.shape)
    # Reserve 10,000 samples for validation
    x_val = x_train[-100:]
    y_val = y_train[-100:]
    x_train = x_train[:-100]
    y_train = y_train[:-100]
    print(x_train[:3])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.mean_squared_error,
        # List of metrics to monitor
        metrics=[keras.metrics.mean_squared_error],
    )
    print("Fit model on training data")
    # print(x_train[:10])
    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=5,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )

    loss, rmse = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test RMSE: {round(rmse, 3)}")
    print(f"Test loss: {round(loss, 3)}")

    preds = model.predict(x_test)
    print("Preds" + str(preds))
    diff = preds.flatten() - y_test
    abs_percDiff = np.abs(diff)
    print('diff = ', diff, 'abs_diff = ', abs_percDiff)
    plt.plot(history.history['mean_squared_error'], label='train')
    plt.plot(history.history['val_mean_squared_error'], label='test')
    plt.show()


if __name__ == '__main__':
    size = 20000

    # processData(size)

    # XGB(size=size)
    NN(size=size)
