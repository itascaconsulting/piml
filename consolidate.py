import numpy as np
import pandas as pd
import matplotlib; matplotlib.rcParams["savefig.directory"] = "."
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})

import numpy as np
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


X = np.array(X)
Ym = np.array(Ym)
Yx = np.array(Yx)
Yy = np.array(Yy)
Yf = np.array(Yf)

np.save("X.npy", X)
np.save("Y.npy", Yf)
