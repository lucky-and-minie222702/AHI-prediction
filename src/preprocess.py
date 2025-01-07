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
                sig = resample(sig, 100 * len(sig) // 128)  # down from 128 to 100 hz
                sig = nk.ecg.ecg_clean(sig, sampling_rate=100)  #  clean
                sig = sig[:total_time * 3000:]  # convert to 30 seconds divisible for total_time
                max_ECG_len = max(max_ECG_len, len(sig))
            else:  # SpO2
                sig = resample(sig, 1 * len(sig) // 8)  # down from 8 to 1 hz
                sig /= 100  # from 0 -> 1
                sig = sig[:total_time * 30:]  # convert to 30 seconds divisible for total_time
                max_SpO2_len = max(max_SpO2_len, len(sig))
            signals[channel] = sig
            
            # start time to label
            start_sleep = str(edf_file.getStartdatetime())[11:]
            
        for key, value in signals.items():
            np.save(path.join("patients", f"patients_{i+1}_{key}"), value)

    except Exception: 
        print(f"Fail to preprocess patient {i+1}")
        edf_file.close()
        exit()
    
    edf_file.close()
    
    content = open(path.join("database", f"{records[i]}_respevt.txt"), "r").readlines()[3:-1:]
    
    ah = list(map(lambda x: x.split()[1], content))
    apnea_count = sum([1 for x in ah if x.startswith("APNEA")])
    hyponea_count = sum([1 for x in ah if x.startswith("HYP")])
    apnea_hyponea_count = apnea_count + hyponea_count
    
    AHI = apnea_hyponea_count / (sleep_time / 3600)
    AI = apnea_count / (sleep_time / 3600)
    HI = hyponea_count / (sleep_time / 3600)
    print(" | => AHI:", AHI)
    print(" | => Apnea index:", AI)
    print(" | => Hyponea index:", HI)
    
    f = open(path.join("patients", f"patients_{i+1}_AHI.txt"), "w")
    print(AHI, file=f)
    print(AI, file=f)
    print(HI, file=f)
    f.close()
    
    AHIs.append(AHI)
    
    # ANNOTATION for each 10 seconds (not timestep)

    # parse times
    content = open(path.join("database", f"{records[i]}_respevt.txt"), "r").readlines()[3:-1:]
    time = list(map(lambda x: x[:8:], content))
    time = list(map(lambda x: calc_time(start_sleep, x), time))
    duration = list(map(lambda x: ith_int(x, 1)[1], [x.split() for x in content]))
    
    annotations = []
    idx = 0
    enough = False
    for t in range(5, total_time * 30 + 5, 5):
        if not enough:
            if t - (time[idx] + duration[idx]) > 2:  # at least 3 / 5 seconds
                idx += 1
                if idx == len(time):
                    annotations.append(0)
                    enough = True
                    continue
            
            if t - time[idx] >= 3:  # at least 3 / 5 seconds
                annotations.append(1)
            else:
                annotations.append(0)
        else:
            annotations.append(0)
    annotations = np.array(annotations)
    print(annotations.shape)
    annotations = np.array(np.split(annotations, len(annotations) // 6))
    annotations = np.round(np.mean(annotations, axis=1))
    print(annotations.shape)
    sleep_stages = list(map(lambda x: 1 if x == 0 else 0, sleep_stages))

    np.save(path.join("patients", f"patients_{i+1}_anns"), annotations)
    np.save(path.join("patients", f"patients_{i+1}_stages"), sleep_stages)


print("\nMax ECG sequence lenght:", max_ECG_len)
print("Max SpO2 sequence lenght:", max_SpO2_len, "\n")
AHIs = np.array(AHIs)
print("Mean AHI:", np.mean(AHIs), "\nMedian AHI:", np.median(AHIs),  "\nStandard deviation AHI:", np.std(AHIs), "\nMax AHI:", np.max(AHIs), "\nMin AHI:", np.min(AHIs))