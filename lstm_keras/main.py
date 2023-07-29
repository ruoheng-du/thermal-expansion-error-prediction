import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from math import sqrt
from numpy import concatenate


def cs_to_sl():
    ## load dataset
    dataset = pd.read_csv('thermal expansion error.csv', header=0, index_col=False)
    dataset.drop(dataset.tail(1).index, inplace=True)
    values = dataset.values
    ## ensure all data is float
    values = values.astype('float32')

    ## split into train and test sets 90:100
    n_train = int(len(dataset) * 0.9)
    train = values[:n_train, :]
    test = values[n_train:, :]

    ## normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)

    return scaled_train, scaled_test, scaler


def train_test(scaled_train, scaled_test):
    ## split into input and outputs
    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    ## reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


def fit_network(train_X, train_y, test_X, test_y, scaler):
    ## def network
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    ## fit network
    history = model.fit(train_X, train_y, epochs=5000, batch_size=512, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    ## plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    ## evaluate the model and calculate Train RMSE
    train_yhat = model.predict(train_X)
    train_yhat_copies_array = np.repeat(train_yhat, 18, axis=-1)
    inv_train_yhat = scaler.inverse_transform(np.reshape(train_yhat_copies_array, (len(train_yhat), 18)))[:, 0]
    train_original_copies_array = np.repeat(train_y, 18, axis=-1)
    inv_train_y = scaler.inverse_transform(np.reshape(train_original_copies_array, (len(train_y), 18)))[:, 0]
    train_rmse = sqrt(mean_squared_error(inv_train_yhat, inv_train_y))
    print('Train RMSE: %.3f' % train_rmse)

    ## plot and check
    pyplot.plot(inv_train_y, color='red', label='Real')
    pyplot.plot(inv_train_yhat, color='green', label='Model')
    pyplot.title('Train Dataset: Thermal Expansion Error Prediction')
    pyplot.xlabel('Time')
    pyplot.ylabel('delta L')
    pyplot.legend()
    pyplot.show()

    ## make a prediction and calculate Test RMSE
    yhat = model.predict(test_X)
    yhat_copies_array = np.repeat(yhat, 18, axis = -1)
    inv_yhat = scaler.inverse_transform(np.reshape(yhat_copies_array, (len(yhat), 18)))[:, 0]
    original_copies_array = np.repeat(test_y, 18, axis=-1)
    inv_y = scaler.inverse_transform(np.reshape(original_copies_array, (len(test_y), 18)))[:, 0]
    rmse = sqrt(mean_squared_error(inv_yhat, inv_y))
    print('Test RMSE: %.3f' % rmse)

    ## plot predicted vs actual
    pyplot.plot(inv_y, color='red', label='Original')
    pyplot.plot(inv_yhat, color='blue', label='Predicted')
    pyplot.title('Thermal Expansion Error Prediction')
    pyplot.xlabel('Time')
    pyplot.ylabel('delta L')
    pyplot.legend()
    pyplot.show()

    ## save model
    model.save('best_model.h5')

if __name__ == '__main__':
    scaled_train, scaled_test, scaler = cs_to_sl()
    train_X, train_y, test_X, test_y = train_test(scaled_train, scaled_test)
    fit_network(train_X, train_y, test_X, test_y, scaler)
