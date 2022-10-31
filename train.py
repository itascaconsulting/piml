import matplotlib; matplotlib.rcParams["savefig.directory"] = "."
import numpy as np
import pylab as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

X = np.load("X.npy")
Y = np.load("Y.npy")
scaler = skl.preprocessing.StandardScaler()
yscaler = skl.preprocessing.StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
scaler.fit(x_train)
y_train = yscaler.fit_transform(y_train)
y_test = yscaler.transform(y_test)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


mlpr = MLPRegressor(
    hidden_layer_sizes=(1000,1000,800,700,600,500),
    activation='tanh',
    solver='sgd',
    learning_rate='adaptive',
    learning_rate_init=0.01,
    momentum=0.90,
    max_iter= 160,
    tol=2.5e-5,
    n_iter_no_change=4,
    random_state=1,
    shuffle=True,
    verbose=True
)

mlpr.fit(x_train_scaled, y_train)
y_pred = mlpr.predict(x_test_scaled)
ts = mlpr.score(x_train_scaled, y_train)
vs = mlpr.score(x_test_scaled, y_test)
