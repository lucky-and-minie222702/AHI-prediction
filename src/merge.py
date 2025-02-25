from data_functions import *
import sys
from os import path
from sklearn.preprocessing import MinMaxScaler

info = open(path.join("data", "info.txt"), "r").readlines()
num_p = int(info[0])
p_list = list(range(1, num_p+1))

seg_len = 30
ecgs = []
spo2s = []
labels = []
rpa = []
rri = []

for p in p_list:
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_spo2 = np.load(path.join("data", f"benhnhan{p}spo2.npy"))
    raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
    # raw_label = raw_label[10:-10:]

    sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=100)
    spo2 = divide_signal(raw_spo2, win_size=(seg_len+1), step_size=1)
    label = divide_signal(raw_label, win_size=(seg_len+1), step_size=1)

    ecgs.append(sig)
    spo2s.append(spo2)
    labels.append(label)

scaler = MinMaxScaler()
ecgs = np.vstack(ecgs)
spo2s = np.vstack(spo2s)
spo2s = spo2 / 100

ecgs = scaler.fit_transform(ecgs.T).T
print(ecgs.nbytes)
ecgs = np.array([clean_ecg(e) for e in ecgs])
# rpa, rri = calc_ecg(ecgs, splr=100, duration=seg_len+1)

full_labels = np.vstack(labels)
print(f"Total samples: {len(full_labels)}")

np.save(path.join("gen_data", "merged_ecg"), ecgs)
# np.save(path.join("gen_data", "merged_rpa"), rpa)
# np.save(path.join("gen_data", "merged_rri"), rri)

np.save(path.join("gen_data", "merged_spo2"), spo2s)
np.save(path.join("gen_data", "merged_label"), full_labels)