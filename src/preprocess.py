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

def is_ahi(lst):
    try:
        a = int(lst[0])
        h = int(lst[1])
    except:
        print(lst)
        exit()
    return a + h > 0

p_list = open(path.join("database", "benhnhanlist.txt"), "r").readlines()[2::]

info_file = open(path.join("data", "info.txt"), "w")
p_list = list(map(lambda x: x.split(), p_list))
for order, name, ahi in p_list:
    no_spo2 = False
    if "*" in order:
        order = order[:-1:]
        no_spo2 = True
    order = int(order)
    print(ahi, file=info_file)
info_file.close()

for order, name, ahi in p_list:
    no_spo2 = False
    if "*" in order:
        order = order[:-1:]
        no_spo2 = True
    order = int(order)
    # buffer
    buffer = 300  # in seconds
    start_time = []
    
    print(f"Patient {order}:")
    
    # label
    content = open(path.join("database", f"benhnhan{order}label.txt")).readlines()[2::]
    start_time.append(content[0][:8:])
    content = list(map(lambda x: x.split() + (["Nothing"] if len(x.split()) == 4 else []), content))
    content = list(map(lambda x: x[2::], content))
    wake = [1 if x[-1] == "Wake" else 0 for x in content]
    wake = np.array(wake)
    
    ahi_events = [is_ahi(x) for x in content]
    ahi_events = np.array(ahi_events)
    
    label = np.stack([ahi_events, wake], axis=1)
    
    # label = content
    print(" => Loading Label")
    
    # ecg
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
    print(" => Loading ECG")
    
    # hr
    content = open(path.join("database", f"benhnhan{order}hr.txt")).readlines()[2::]
    start_time.append(content[0][:8:])
    content = list(map(lambda x: float(x.split()[-1]) if len(x.split()) == 3 else None, content))
    content = set_prior_none_to_zero(content)
    content = fill_missing_with_mean(content)
    content = np.array(content)
    hr = content
    print(" => Loading HR")
    
    # spo2
    spo2 = None
    if no_spo2:
        print(" => No SpO2!")
        sys.stdout.flush()
    else:
        content = open(path.join("database", f"benhnhan{order}spo2.txt")).readlines()[2::]
        start_time.append(content[0][:8:])
        content = list(map(lambda x: float(x.split()[-1]) if len(x.split()) == 3 else None, content))
        content = set_prior_none_to_zero(content)
        content = fill_missing_with_mean(content)
        content = np.array(content)
        spo2 = content
        print(" => Loading SpO2")
    
    sigs = [label, ecg, hr]
    if spo2 is not None:
        sigs.append(spo2)
    sig_labels = ["Label", "ECG", "HR", "SpO2"]
    sig_splr = [1, 100, 1, 1]
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
    
    print(" => Processing...")
    
    if not all(len(sigs[0]) // sig_splr[0] == len(sigs[i]) // sig_splr[i] for i in range(len(sigs))):
        print(f" => Failed!")
        for i in range(len(sigs)):
            print(f"{sig_labels[i]}: {len(sigs[i])}", end = " ")
        print()
    else:
        # sigs[0] = np.transpose(sigs[0], (1, 0))
        for i in range(len(sigs)):
            np.save(path.join("data", f"benhnhan{order}{sig_labels[i].lower()}"), sigs[i])
        print(f" => Successful!")