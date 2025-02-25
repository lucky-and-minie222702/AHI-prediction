import enum
import numpy as np
from os import path
import neurokit2 as nk
from scipy import signal
from data_functions import *
import sys

def set_prior_none_to_zero(lst: list):
    for i in range(len(lst)):
        if lst[i] is not None:
            break
        lst[i] = 0
    return lst

p_list = open(path.join("database", "benhnhanlist.txt"), "r").readlines()[1::]
p_list = list(map(lambda x: x.split(), p_list))

male_count = 0
female_count = 0
bmis = []
ahis = []
dur_abnormal = []
dur_normal = []

# Order Name AHI Sex BMI
for order, name, ahi, sex, bmi in p_list:
    if sex == "M":
        male_count += 1
    else:
        female_count +=1
    ahis.append(float(ahi))
    bmis.append(float(bmi))
    
    order = int(order)
    # buffer
    buffer = 300  # in seconds
    start_time = []

    print(f"Patient {order} - {name}:")
    
    # label
    content = open(path.join("database", f"benhnhan{order}label.txt")).readlines()[2::]
    
    print(" => Loading Label")
    start_scoring = -1
    for s in content:
        start_scoring += 1
        if len(s.split()) == 4:
            break
    end_scoring = len(content) + 1
    for s in content[::-1]:
        end_scoring -= 1
        if len(s.split()) == 4:
            break
    content = content[start_scoring:end_scoring:]

    start_time.append(content[0][:8:])
    content = list(map(lambda x: x.split(), content))
    content = list(map(lambda x: x[2::], content))
    
    for idx, _ in enumerate(content):
        if len(content[idx]) < 2:
            content[idx].append(content[idx-1][1])

    wake = [1 if x[1] == "Wake" else 0 for x in content]
    wake = np.array(wake)

    ahi_events = [0 if int(x[0]) == 0 else 1 for x in content]
    ahi_events = np.array(ahi_events)
    
    label = np.stack([ahi_events, wake], axis=1)
    
    # ecg
    print(" => Loading ECG...")
    content = open(path.join("database", f"benhnhan{order}ecg.txt")).readlines()[2::]
    time_ecg = [x[:11:] for x in content]
    content = content[time_ecg.count(time_ecg[0])::]
    start_time.append(content[0][:8:])
    content = list(map(lambda x: float(x.split()[-1]) if len(x.split()) == 3 else None, content))
    content = set_prior_none_to_zero(content)
    content = fill_missing_with_mean(content)
    content = np.array(content)
    # content = nk.ecg.ecg_clean(content, sampling_rate=200)
    content = signal.resample(content, int(len(content) / 200 * 100))  # to 100hz
    ecg = content
    
    print(" => Loading SpO2...")
    content = open(path.join("database", f"benhnhan{order}spo2.txt")).readlines()[2::]
    start_time.append(content[0][:8:])
    content = list(map(lambda x: float(x.split()[-1]) if len(x.split()) == 3 else None, content))
    content = set_prior_none_to_zero(content)
    content = fill_missing_with_mean(content)
    content = np.array(content)
    spo2 = content
        
    print(" => Processing...")
    sigs = [label, ecg, spo2]
    sig_labels = ["Label", "ECG", "SpO2"]
    sig_splr = [1, 100, 1]
    ideal_len = len(label) - buffer * 2

    start_time = list(map(time_to_seconds, start_time))
    ideal_time = min(start_time)
    shift_time = [t - ideal_time for t in start_time]
    for i in range(len(sigs)):
        sigs[i] = sigs[i][shift_time[i]*sig_splr[i]::]
        sigs[i] = sigs[i][buffer*sig_splr[i]:-(buffer*sig_splr[i]):]
        if len(sigs[i]) <= ideal_len*sig_splr[i]:
            zeros_pad = np.zeros([ideal_len*sig_splr[i] - len(sigs[i])] + ([2] if i == 0 else []))
            sigs[i] = np.concatenate([sigs[i], zeros_pad]) 
        else:
            sigs[i] = sigs[i][:ideal_len*sig_splr[i]:]
    
    if not all(len(sigs[0]) // sig_splr[0] == len(sigs[i]) // sig_splr[i] for i in range(len(sigs))):
        print(f" => Status: failed!")
        for i in range(len(sigs)):
            print(f"{sig_labels[i]}: {len(sigs[i])} -", end = " ")
        print()
    else:
        for i in range(len(sigs)):
            if i == 0:
                cg = count_groups(sigs[i][::, :1:].flatten())
                dur_normal.extend(list(map(lambda x: x[1], cg[0])))
                dur_abnormal.extend(list(map(lambda x: x[1], cg[1])))
            np.save(path.join("data", f"benhnhan{order}{sig_labels[i].lower()}"), sigs[i])
        print(f" => Status: successful!")


ahis = np.array(ahis)
bmis = np.array(bmis)
dur_normal = np.array(dur_normal)
dur_abnormal = np.array(dur_abnormal)

info_file = open(path.join("data", "info.txt"), "w")
sys.stdout = Tee(info_file)

np.save(path.join("info", "ahi"), ahis)
np.save(path.join("info", "bmi"), bmis)
np.save(path.join("info", "dur_normal"), dur_normal)
np.save(path.join("info", "dur_abnormal"), dur_abnormal)

print("Mean - Std - Min - Max of:")
print("AHI:", np.mean(ahis), np.std(ahis), np.min(ahis), np.max(ahis))
print("BMI:", np.mean(bmis), np.std(bmis), np.min(bmis), np.max(bmis))
print("Normal-Duration:", np.mean(dur_normal), np.std(dur_normal), np.min(dur_normal), np.max(dur_normal))
print("Abnormal-Duration:", np.mean(dur_abnormal), np.std(dur_abnormal), np.min(dur_abnormal), np.max(dur_abnormal))
print("Sex-(Male-Female):", male_count, female_count)

Tee.reset()