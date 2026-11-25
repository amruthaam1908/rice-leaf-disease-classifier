import pickle
import os
import sys

log_file = 'd:/rice main project/pickle_log.txt'

with open(log_file, 'w') as log:
    try:
        path = 'd:/rice main project/models/class_indices.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            log.write(f"Data: {data}\n")
            print(f"Data: {data}")
        else:
            log.write("File not found\n")
            print("File not found")
    except Exception as e:
        log.write(f"Error: {e}\n")
        print(f"Error: {e}")
