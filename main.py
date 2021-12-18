import math

import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNClassification

# load data
data_path = "sunspots_2900.txt"
data = np.loadtxt(data_path, dtype=float)

plt.plot(data)
plt.show()

# describle
print("Count: ", len(data))
print("Mean: ", data.mean())
print("Std: ", data.std())
print("Min: ", data.min())
print("Max: ", data.max())

data = data.reshape(-1, 1)
print("Dataset shape: ", data.shape)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# train test split
no_data = len(data)
ratio = 0.8
train = data[:int(no_data*ratio)]
test = data[int(no_data*ratio):]
print(len(train))
print(len(test))

# reshape into X=t and Y=t+1


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, features]
trainX = np.squeeze(trainX, axis=2)
testX = np.squeeze(testX, axis=2)
trainY = np.squeeze(trainY, axis=1)
testY = np.squeeze(testY, axis=1)

# Training
model = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                    learning_rate_rbm=0.05,
                                    learning_rate=0.1,
                                    n_epochs_rbm=10,
                                    n_iter_backprop=100,
                                    batch_size=32,
                                    activation_function='relu',
                                    dropout_p=0.2)

model.fit(trainX, trainY)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# metric RMSE
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# save
np.savetxt('trainPredict.txt', trainPredict)
np.savetxt('testPredict.txt', testPredict)
np.savetxt('trainY.txt', trainY)
np.savetxt('testY.txt', testY)
