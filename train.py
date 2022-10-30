import matplotlib; matplotlib.rcParams["savefig.directory"] = "."
import numpy as np
import pylab as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

X = np.load("X.npy")
Y = np.load("Y.npy")

Y = Y[:,89]
scaler = skl.preprocessing.StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

mlpr = MLPRegressor(
    hidden_layer_sizes=(20,20,20),
    activation='tanh',
    solver='sgd',
    learning_rate='adaptive',
    learning_rate_init=0.01,
    momentum=0.90,
    max_iter=10 * 160,
    tol=2.5e-5,
    n_iter_no_change=10,
    random_state=1,
    shuffle=True,
    verbose=True
)

mlpr.fit(x_train_scaled, y_train)
y_pred = mlpr.predict(x_test_scaled)
ts = mlpr.score(x_train_scaled, y_train)
vs = mlpr.score(x_test_scaled, y_test)
