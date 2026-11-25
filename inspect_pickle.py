import pickle
import os

path = 'd:/rice main project/models/class_indices.pkl'
if os.path.exists(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(data)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("File not found")
