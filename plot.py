# %%
import numpy as np
import matplotlib.pyplot as plt

# load data
data_path = "data.txt"
data = np.loadtxt(data_path, dtype=float)

trainPredict = np.loadtxt("trainPredict.txt", dtype=float)
trainY = np.loadtxt("trainY.txt", dtype=float)
testPredict = np.loadtxt("testPredict.txt", dtype=float)
testY = np.loadtxt("testY.txt", dtype=float)

plt.plot(trainPredict[:200], 'b', label="DBN Predict")
plt.plot(trainY[:200], 'r', label="Train Data")
plt.legend(loc="upper right")
plt.show()

# %%
plt.plot(testPredict[:200], 'g', label="DBN Predict")
plt.plot(testY[:200], 'r', label="Test Data")
plt.legend(loc="upper left")
plt.show()
