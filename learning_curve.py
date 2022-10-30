import matplotlib; matplotlib.rcParams["savefig.directory"] = "."
import numpy as np
import pylab as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import time
import os
import joblib

start_time = time.time()

def train_and_test(X_train, y_train, X_test, y_test):
    sizes = (500,400,200,100)

    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlpr = MLPRegressor(
        hidden_layer_sizes=sizes,
        activation='relu',
        solver='sgd',
        learning_rate='adaptive',
        learning_rate_init=0.01,
        momentum=0.90,
        max_iter=10 * 160,
        tol=2.5e-5,
        n_iter_no_change=10,
        random_state=1,
        shuffle=True,
        verbose=False
    )

    mlpr.fit(X_train_scaled, y_train)

    y_pred = mlpr.predict(X_test_scaled)

    ts = mlpr.score(X_train_scaled, y_train)
    vs = mlpr.score(X_test_scaled, y_test)
    print(ts, vs)

    return (scaler, mlpr), ts, vs, X_train_scaled, y_train, X_test_scaled, y_test, y_pred

lhc_sizes = range(1, 17)
number_of_unknowns = 4
X_train, X_test = [], []
Y_train, Y_test = [], []
test_score, validation_score = [], []

sizes = []
cube_size = []
for lhc_size in lhc_sizes:
    X = np.load(f"cube_{number_of_unknowns}_{lhc_size}.npy")
    Y = np.load(f"result_cube_{number_of_unknowns}_{lhc_size}.npy")
    print(f"training {lhc_size}")
    sizes.append(len(Y)+sum(cube_size))
    cube_size.append(len(Y))

    #Y = Y.reshape(len(Y), 55*2)
    Y = np.linalg.norm(Y,axis=2)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
    X_train += x_train.tolist()
    X_test += x_test.tolist()
    Y_train += y_train.tolist()
    Y_test += y_test.tolist()

    res = train_and_test(np.array(X_train), np.array(Y_train),
                         np.array(X_test), np.array(Y_test))
    (scaler, mlpr), ts, vs, X_train_scaled, y_train, X_test_scaled, y_test, y_pred = res
    test_score.append(ts)
    validation_score.append(vs)
    print(len(X_train)+len(X_test), vs)
    print(' ')


plt.semilogx(sizes, -np.log10(1-np.array(validation_score)), "o-")
plt.ylabel("Model score [$-log_{10}(1-R^2)$]")
plt.xlabel("Number of samples []")
bbox = dict(boxstyle='round', facecolor='white')
plt.axhline(-np.log10(1-0.9), color="black")
plt.axhline(-np.log10(1-0.99), color="black")
plt.axhline(-np.log10(1-0.999), color="black")
plt.axhline(-np.log10(1-0.9999), color="black")
plt.text(20, 1.0, "$R^2=0.9$", bbox=bbox, verticalalignment="center")
plt.text(20, 2.0, "$R^2=0.99$", bbox=bbox, verticalalignment="center")
plt.text(20, 3.0, "$R^2=0.999$", bbox=bbox, verticalalignment="center")
plt.ylim(None, 3.3)
plt.xlim(None, 1e6)
plt.show()
