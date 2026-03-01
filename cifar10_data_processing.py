import pandas as pd
import numpy as np
import os
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_batches = []

for i in range(1, 6):
    data = unpickle(f"cifar-10-batches-py/data_batch_{i}")
    file_names = []
    for file_name, np_array in zip(data[b"filenames"], list(data[b"data"])):
        file_name = f"data/train/{file_name.decode('utf-8').replace('.png', '.npy')}"
        file_names.append(file_name)
        np.save(file_name, np_array)
    

    batch_data = pd.DataFrame({"labels":data[b"labels"],"filenames":file_names})
    train_batches.append(batch_data)

train_df = pd.concat(train_batches)
train_df.to_csv("data/train.csv", index=False)

test_data = unpickle(f"cifar-10-batches-py/test_batch")
file_names = []
for file_name, np_array in zip(test_data[b"filenames"], list(test_data[b"data"])):
    file_name = f"data/test/{file_name.decode('utf-8').replace('.png', '.npy')}"
    file_names.append(file_name)
    np.save(file_name, np_array)

test_df = pd.DataFrame({"labels":test_data[b"labels"],"filenames":file_names})
test_df.to_csv("data/test.csv", index=False)