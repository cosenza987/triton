import numpy as np

a = np.arange(0, 10)
b = np.arange(10, 20)
print(a[:, None] + b[None, :])