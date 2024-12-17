from os import path
import pyedflib
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

scaler = StandardScaler()
max_len = 0
records = [f"ucddb{i:0{3}d}" for i in range(2, 29) if i not in [4, 16]]

for i in range(len(records)):
    print(f"Preprocessing patient {i+1}:")
    edf_file = pyedflib.EdfReader(path.join("database", f"{records[i]}.rec"))
    max_len = max(max_len, edf_file.file_duration // 2)
    
    try:
        channels = ["ECG", "SpO2"]
        signals = {}
        
        for channel in channels:
            idx = edf_file.getSignalLabels().index(channel)
            sig = edf_file.readSignal(idx)
            if channels == "ECG":
                sig = resample(sig, len(sig) // 2)
            signals[channel] = sig
            
        for key, value in signals.items():
            value = scaler.fit_transform(value.reshape(-1, 1)).flatten()
            np.save(path.join("patients", f"patients_{i+1}_{key}"), value)

        print(f"Succeed in preprocessing patient {i+1}")
    except Exception:
        print(f"Fail to preprocess patient {i+1}")
    
    edf_file.close()
    
    sleep_stages = list(map(int, open(path.join("database", f"{records[i]}_stage.txt")).readlines()))
    total_time = len(sleep_stages)
    sleep_time = (total_time - sleep_stages.count(0)) * 30
    print("| Sleep time:", f"{sleep_time} seconds", f" {(sleep_time / 3600):.2f} hours")
    
    hyponea_apnea_count = len(open(path.join("database", f"{records[i]}_respevt.txt"), "r").readlines()) - 3 # -3 labelling and outline lines
    print("| Hyponea apnea count:", hyponea_apnea_count)
    AHI = hyponea_apnea_count / (sleep_time / 3600)
    print("| => AHI:", AHI)
    
    f = open(path.join("patients", f"patients_{i+1}_AHI.txt"), "w")
    print(AHI, file=f)
    f.close()

print("Max sequence lenght:", max_len)