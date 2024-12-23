
import scipy.io
import numpy as np

mat_file_path = 'features_1.mat'  
mat_data = scipy.io.loadmat(mat_file_path)

for key in mat_data.keys():
    print(key)

if 'features' in mat_data:
    data = mat_data['features']
    print("Data shape:", data.shape)
    print("Data content:")

    try:
        print(data)
    except Exception as e:
        print("Error printing data directly:", e)
        print("Data type:", type(data))
