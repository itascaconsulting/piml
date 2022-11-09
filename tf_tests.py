import os

import tensorflow as tf

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.metrics import r2_score

X = []
Ym = []
Yx = []
Yy = []
Yf = []
for i in range(1,18):
  X += np.load(f"cube_4_{i}.npy").tolist()
  Y0 = np.load(f"result_cube_4_{i}.npy")
  Yx += Y0[:,:,0].tolist()
  Yy += Y0[:,:,1].tolist()
  Ym += np.linalg.norm(Y0,axis=2).tolist()
  Yf += Y0.reshape(len(Y0),110).tolist()
scale = np.array(Ym).max()
X = np.array(X)
Ym = np.array(Ym)
Yx = np.array(Yx)
Yy = np.array(Yy)
Yf = np.array(Yf)

# dont predict the columns (individual gridpoint displacement vector components) that we know are zero.
column_mask = Yf[0]==0
zero_cols = np.arange(Yf.shape[-1])[column_mask]
Ynz = Yf[:, ~column_mask]
x_train, x_test, y_train, y_test = train_test_split(X, Ynz, test_size=0.2, random_state = 42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#ysc = StandardScaler()
#y_train = ysc.fit_transform(y_train)
#y_test = ysc.transform(y_test)
y_train /= scale
y_test /= scale

ann1 = tf.keras.models.Sequential()
ann1.add(tf.keras.layers.Dense(units = 1000,activation = 'relu'))
ann1.add(tf.keras.layers.Dense(units = 1000,activation = 'relu'))
ann1.add(tf.keras.layers.Dense(units = 800,activation = 'relu'))
ann1.add(tf.keras.layers.Dense(units = 700,activation = 'relu'))
ann1.add(tf.keras.layers.Dense(units = 600,activation = 'relu'))
ann1.add(tf.keras.layers.Dense(units = 500,activation = 'relu'))
#Output Layer
ann1.add(tf.keras.layers.Dense(units=y_train.shape[-1] , activation = 'linear'))
metric = tfa.metrics.r_square.RSquare()
ann1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=4e-4),
    loss='mean_squared_error',
    metrics=[metric])

ann1.fit(x_train,y_train,batch_size=32, epochs=20)



y_pred = ann1.predict(x_test)
y_actual = y_test
y_pred_scaled = ysc.inverse_transform(y_pred)
y_actual_scaled = ysc.inverse_transform(y_actual)
print("test set score", r2_score(y_pred, y_actual))
ann1.save('piml_test.h5')

# add back the zero columns
for i in zero_cols:
  y_pred_scaled = np.insert(y_pred_scaled, i, 0, axis=1)
  y_actual_scaled = np.insert(y_actual_scaled, i, 0, axis=1)
# this is the gridpoint order that puts the gridpoints into an array order (From np.lexsort(gpa.pos()))
order = np.array([ 0,  1,  4,  6,  8, 10, 12, 14, 16, 18, 20,  2,  3,  5,  7,  9, 11,
       13, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54], dtype=np.int64)
1/0
case = 0
ya = y_actual_scaled[case].reshape(55,2)[order]
yamag = np.linalg.norm(ya,axis=1)
yp = y_pred_scaled[case].reshape(55,2)[order]
ypmag = np.linalg.norm(yp,axis=1)
fig, ax1 = plt.subplots()
im1 = ax1.imshow(yamag.reshape(5,11))
#im1 = ax1.imshow(a, interpolation='nearest', aspect=1)
fig.colorbar(im1)
plt.title("Displacement Magnitude - Predicted")
fig, ax2 = plt.subplots()
im2 = ax2.imshow(ypmag.reshape(5,11))
fig.colorbar(im2)
plt.title("Displacement Magnitude - Actual")
plt.show()

np.save("tc_X.npy", sc.inverse_transform(x_test)[0])
np.save("tc_Yp.npy", y_pred_scaled[0])
np.save("tc_Ya.npy", y_actual_scaled[0])
dump(sc, "sc.pkl")
dump(ysc, "ysc.pkl")
