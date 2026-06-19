import numpy as np

data_points = np.array([10, 20, 30, 40])

matrix = data_points.reshape(2, 2)
print("Matrix:\n", matrix)

processed_data = data_points * 2
print("Scaled Data:", processed_data)