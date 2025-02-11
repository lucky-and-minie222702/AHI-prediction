from data_functions import *
import sys
from os import path
from sklearn.preprocessing import MinMaxScaler

info = open(path.join("data", "info.txt"), "r").readlines()
p_list = []
no_spo2 = []

for s in info:
    s = s[:-1:]
    if "*" in s:
        p_list.append(int(s[1::]))
        no_spo2.append(int(s[1::]))
    else:
        p_list.append(int(s))

p_list = [x for x in p_list if x not in no_spo2]
num_p = len(p_list)

seg_len = 10
ecgs = []
spo2s = []
labels = []
rpa = []
rri = []

for p in p_list:
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_spo2 = np.load(path.join("data", f"benhnhan{p}spo2.npy"))
    raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
    raw_label = raw_label[10:-10:]

    sig = divide_signal(raw_sig, win_size=(10+seg_len+10)*100, step_size=(seg_len*100) // 2)
    spo2 = divide_signal(raw_spo2, win_size=(10+seg_len+10), step_size=seg_len // 2)
    label = divide_signal(raw_label, win_size=seg_len, step_size=seg_len // 2)

    ecgs.append(sig)
    spo2s.append(spo2)
    labels.append(label)

scaler = MinMaxScaler()
ecgs = np.vstack(ecgs)
spo2s = np.vstack(spo2s)
spo2s = scaler.fit_transform(spo2s.T).T


# augment
ecgs = scaler.fit_transform(ecgs.T).T
ecgs = np.array([nk.ecg.ecg_clean(e, sampling_rate=100, method="pantompkins1985") for e in ecgs])
rpa, rri = calc_ecg(ecgs, splr=100, duration=10+seg_len+10)


full_labels = np.vstack(labels)

np.save(path.join("gen_data", "merged_ecg"), ecgs)
np.save(path.join("gen_data", "merged_spo2"), spo2s)
np.save(path.join("gen_data", "merged_rpa"), rpa)
np.save(path.join("gen_data", "merged_rri"), rri)
np.save(path.join("gen_data", "merged_label"), labels)