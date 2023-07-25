# from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
# from numpy import concatenate
from math import sqrt


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    ## convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def cs_to_sl():
    ## load dataset
    dataset = pd.read_csv('Thermal Expansion Error.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]], axis=1, inplace=True)
    print(reframed.head())
    return reframed, scaler


def train_test(dataset):
    # split into train and test sets 90:100
    values = dataset.values
    n_train = int(len(dataset) * 0.9)
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


def fit_network(train_X, train_y, test_X, test_y, scaler):
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    ## fit network
    history = model.fit(train_X, train_y, epochs=400, batch_size=32, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    ## plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    ## make a prediction
    yhat = model.predict(test_X)
    yhat_copies_array = np.repeat(yhat, 16, axis = -1)
    predicted = scaler.inverse_transform(np.reshape(yhat_copies_array, (len(yhat), 16)))[:, 0]
    # print(predicted)
    original_copies_array = np.repeat(test_y, 16, axis=-1)
    original = scaler.inverse_transform(np.reshape(original_copies_array, (len(test_y), 16)))[:, 0]
    # print(original)
    ## calculate RMSE
    rmse = sqrt(mean_squared_error(predicted, original))
    print('Test RMSE: %.3f' % rmse)
    ## plot predicted vs actual
    pyplot.plot(original, color='red', label='Actual')
    pyplot.plot(predicted, color='blue', label='Predicted')
    pyplot.title('Thermal Expansion Error Prediction')
    pyplot.xlabel('Time')
    pyplot.ylabel('delta L')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    dataset, scaler = cs_to_sl()
    train_X, train_y, test_X, test_y = train_test(dataset)
    fit_network(train_X, train_y, test_X, test_y, scaler)

