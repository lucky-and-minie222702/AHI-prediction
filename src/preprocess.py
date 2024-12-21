from os import path
import pyedflib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import resample
from data_functions import *
import neurokit2 as nk

scaler = MinMaxScaler()
max_ECG_len = 0
max_SpO2_len = 0
records = [f"ucddb{i:0{3}d}" for i in range(2, 29) if i not in [4, 16]]
AHIs = []

for i in range(len(records)):
    print(f"Preprocessing patient {i+1}:")
    edf_file = pyedflib.EdfReader(path.join("database", f"{records[i]}.rec"))

    try:
        channels = ["ECG", "SpO2"]
        signals = {}
        
        sleep_stages = list(map(int, open(path.join("database", f"{records[i]}_stage.txt")).readlines()))
        total_time = len(sleep_stages)
        sleep_time = (total_time - sleep_stages.count(0)) * 30  # in seconds
        print(" | Sleep time:", f"{sleep_time} seconds", f" {(sleep_time / 3600):.2f} hours")
        
        for channel in channels:
            idx = edf_file.getSignalLabels().index(channel)
            sig = edf_file.readSignal(idx)
            if channel == "ECG":
                sig = sig[:total_time*30*128:] 
                sig = resample(sig, 100 * len(sig) // 128)  # down from 128 to 100 hz
                sig = sig[:len(sig) // 1000 * 1000:]  # convert to 10 seconds divisible
                sig = nk.ecg.ecg_clean(sig, sampling_rate=100)  #  clean
                max_ECG_len = max(max_ECG_len, len(sig))
            else:  # SpO2
                sig = sig[:total_time*30*8:]
                sig = resample(sig, 1 * len(sig) // 8)  # down from 8 to 1 hz
                sig = sig[:len(sig) // 10 * 10:]  # convert to 10 seconds divisible
                sig /= 100  # from 0 -> 1
                max_SpO2_len = max(max_SpO2_len, len(sig))
            signals[channel] = sig
            
        for key, value in signals.items():
            # value = scaler.fit_transform(value.reshape(-1, 1)).flatten()
            np.save(path.join("patients", f"patients_{i+1}_{key}"), value)

        print(f"Succeed in preprocessing patient {i+1}")
    except Exception: 
        print(f"Fail to preprocess patient {i+1}")
        exit()
    
    edf_file.close()
    
    apnea_hyponea_count = len(open(path.join("database", f"{records[i]}_respevt.txt"), "r").readlines()) - 4 # -4 labelling and outline lines
    print(" | Hyponea apnea count:", apnea_hyponea_count)
    AHI = apnea_hyponea_count / (sleep_time / 3600)
    print(" | => AHI:", AHI)
    
    f = open(path.join("patients", f"patients_{i+1}_AHI.txt"), "w")
    print(AHI, file=f)
    f.close()
    
    AHIs.append(AHI)
    
    # ANNOTATION for each 10 seconds (not timestep)
    
    content = open(path.join("database", f"{records[i]}_respevt.txt"), "r").readlines()[3:-1:]
    time = list(map(lambda x: x[:8:], content))
    time = list(map(time_to_seconds, time))
    duration = list(map(lambda x: ith_int(x, 1)[1], [x.split() for x in content]))
    
    annotations = []
    idx = 0
    enough = False
    for t in range(0, total_time * 30, 10):
        if not enough:
            if t >= time[idx]:
                annotations.append(1)
            else:
                annotations.append(0)

            duration10s = t - time[idx] 

            if  duration10s >= duration[idx] and duration10s - duration[idx] > 3:  # at least 7 / 10 seconds
                idx += 1
                if idx == len(time):
                    enough = True
        else:
            annotations.append(0)

    annotations = np.array(annotations)
    sleep_stages = list(map(lambda x: 1 if x == 0 else 0, sleep_stages))
    sleep_stages_10s = np.array([i for i in sleep_stages for _ in range(3)])
    np.save(path.join("patients", f"patients_{i+1}_anns"), annotations)
    np.save(path.join("patients", f"patients_{i+1}_stages"), sleep_stages_10s)


print("\nMax ECG sequence lenght:", max_ECG_len)
print("Max SpO2 sequence lenght:", max_SpO2_len, "\n")
AHIs = np.array(AHIs)
print("Mean AHI:", np.mean(AHIs), "\nMedian AHI:", np.median(AHIs),  "\nStandard deviation AHI:", np.std(AHIs), "\nMax AHI:", np.max(AHIs), "\nMin AHI:", np.min(AHIs))