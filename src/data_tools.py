from data_functions import *
import sys
from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

if len(sys.argv) <  2:
    print("No command, exit!")
    exit()

if sys.argv[1] == "merge":
    stages = []
    annotations = []
    sequences_ECG = []
    sequences_SpO2 = []

    for i in range(1, 26):
        seq_ECG = np.load(path.join("patients", f"patients_{i}_ECG.npy"))
        seq_SpO2 = np.load(path.join("patients", f"patients_{i}_SpO2.npy"))
        ann = np.load(path.join("patients", f"patients_{i}_anns.npy"))
        stage = np.load(path.join("patients", f"patients_{i}_stages.npy"))
        
        sequences_ECG.extend(seq_ECG.tolist())  # 60 seconds
        sequences_SpO2.extend(seq_SpO2.tolist()) 
        
        annotations.extend(ann.tolist())
        stages.extend(stage.tolist())

    sequences_ECG = np.array(sequences_ECG)
    sequences_SpO2 = np.array(sequences_SpO2)
    annotations = np.array(annotations)
    stages = np.array(stages)
    
    sequences_ECG = divide_signal(sequences_ECG, win_size=6000, step_size=500)  # 30s, step 5s
    sequences_SpO2 = divide_signal(sequences_SpO2, win_size=60, step_size=5)  # 30s, step 5s
    annotations = divide_signal(annotations, win_size=60, step_size=5)
    stages = divide_signal(stages, win_size=60, step_size=5)
    annotations = np.array(
        [1 if np.count_nonzero(x) == 10 else 0 for x in annotations]
    )
    stages = np.array(
        [1 if np.count_nonzero(x) == 10 else 0 for x in stages]
    )

    
    sequences_ECG = scaler.fit_transform(sequences_ECG.T).T  # scale
    
    rpa, rri = calc_ecg(sequences_ECG)
    best_ecg = np.count_nonzero(rpa, axis=1) >= 45  # min 45 bpm
    best_spo2 = np.min(sequences_SpO2, axis=1) >= 0.8
    best = np.array([e and s for e, s in zip(best_ecg, best_spo2)])
    
    sequences_ECG = sequences_ECG[best]
    sequences_SpO2 = sequences_SpO2[best]
    annotations = annotations[best]
    stages = stages[best]
    
    # augment
    sequences_ECG = np.vstack(
        [sequences_ECG, sequences_ECG + np.random.normal(0.0, 0.005, sequences_ECG.shape), add_baseline_wander(sequences_ECG, frequency=0.05, amplitude=0.05, sampling_rate=100, flat_rate=0.5)]
    )
    sequences_SpO2 = np.vstack(
        [sequences_SpO2, sequences_SpO2 + np.random.normal(0.0, 0.0025, sequences_SpO2.shape), sequences_SpO2 + np.random.normal(0.0, 0.005, sequences_SpO2.shape)]
    )
    
    annotations = np.concatenate([annotations, annotations, annotations])
    stages = np.concatenate([stages, stages, stages])
    
    _, counts = np.unique(annotations, return_counts=True)
    print(f"Annotations: [0]: {counts[0]} | [1]: {counts[1]}")
    _, counts = np.unique(stages, return_counts=True)
    print(f"Stages: [0]: {counts[0]} | [1]: {counts[1]}")

    np.save(path.join("patients", "merged_ECG"), sequences_ECG)
    np.save(path.join("patients", "merged_SpO2"), sequences_SpO2)
    np.save(path.join("patients", "merged_anns"), annotations)
    np.save(path.join("patients", "merged_stages"), stages)