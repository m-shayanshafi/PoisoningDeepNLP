import pickle
import numpy as np

def convert_to_pickle(item, directory):
    pickle.dump(item, open(directory,"wb"))

def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))

def max_length(tensor):
    return max(len(t) for t in tensor)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

