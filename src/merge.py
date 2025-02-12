from data_functions import *
import sys
from os import path
from sklearn.preprocessing import MinMaxScaler

info = open(path.join("data", "info.txt"), "r").readlines()
p_list = []
no_spo2 = []
heavy_loading = "heavy" in sys.argv:

for s in info:
    s = s[:-1:]
    if "*" in s:
        p_list.append(int(s[1::]))
        no_spo2.append(int(s[1::]))
    else:
        p_list.append(int(s))

p_list = [x for x in p_list if x not in no_spo2]
num_p = len(p_list)

seg_len = 30
ecgs = []
spo2s = []
labels = []
rpa = []
rri = []

for p in p_list:
    if heavy_loading:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_spo2 = np.load(path.join("data", f"benhnhan{p}spo2.npy"))
    raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
    # raw_label = raw_label[10:-10:]

    if heavy_loading:
        sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=100)
    spo2 = divide_signal(raw_spo2, win_size=(seg_len+1), step_size=1)
    label = divide_signal(raw_label, win_size=(seg_len+1), step_size=1)

    if heavy_loading:
        ecgs.append(sig)
    spo2s.append(spo2)
    labels.append(label)

scaler = MinMaxScaler()
if heavy_loading:
    ecgs = np.vstack(ecgs)
spo2s = np.vstack(spo2s)
spo2s = spo2 / 100

print(f"Total samples: {len(labels)}")

if heavy_loading:
    ecgs = scaler.fit_transform(ecgs.T).T
    print(ecgs.nbytes)
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    rpa, rri = calc_ecg(ecgs, splr=100, duration=seg_len+1)

full_labels = np.vstack(labels)

if heavy_loading:
    np.save(path.join("gen_data", "merged_ecg"), ecgs)
    np.save(path.join("gen_data", "merged_rpa"), rpa)
    np.save(path.join("gen_data", "merged_rri"), rri)

np.save(path.join("gen_data", "merged_spo2"), spo2s)
np.save(path.join("gen_data", "merged_label"), full_labels)