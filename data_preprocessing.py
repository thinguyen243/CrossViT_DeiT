import os
import torch
import pickle
import numpy as np

folder_path = "/home/thynnl/Downloads/Imagenet8_train"

def read_file(filepath):
    try:
        datada = torch.load(filepath )
        print(type(datada))
        return datada
    except Exception as e:
        pass
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"Đọc được bằng numpy.load() — Loại: {type(data)}")
        return data
    except Exception as e:
        pass
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            print(f"Đọc được bằng pickle.load() — Loại: {type(data)}")
            return data
    except Exception as e:
        print(f"Không đọc được file — {e}")
    return None

folder_data = []
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    print(f"{filename}")
    if os.path.isfile(filepath):
        folder_data.append(read_file(filepath))
print(folder_data)