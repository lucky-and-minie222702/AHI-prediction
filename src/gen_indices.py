import numpy as np
from sklearn.model_selection import KFold
from os import path

info = open(path.join("data", "info.txt"), "r").readlines()
p_list = []
no_spo2 = []

for s in info:
    s = s[:-1:]
    if "*" in s:
        no_spo2.append(int(s[1::]))
    else:
        p_list.append(int(s))

# p_list = [x for x in p_list if x not in no_spo2]
p_list = np.array(p_list)
num_p = len(p_list)

kf = KFold(n_splits=3, shuffle=True, random_state=np.random.randint(22022009))

idx = 1

for i, (train_index, test_index) in enumerate(kf.split(p_list)):
    np.save(path.join("gen_data", f"fold_{i}_train"), p_list[train_index])
    np.save(path.join("gen_data", f"fold_{i}_test"), p_list[test_index])