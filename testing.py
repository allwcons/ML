import numpy as np
y = np.array([9,8,7,6])
x = np.argsort(y)[:3]
print(y[x])