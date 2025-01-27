import numpy as np
from os import path
from sklearn.model_selection import train_test_split

annotations  = np.load(path.join("patients", "merged_anns.npy"))
indices = np.arange(len(annotations))

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(69696969))
np.save(path.join("patients", "train_indices_ah"), train_indices)
np.save(path.join("patients", "test_indices_ah"), test_indices)