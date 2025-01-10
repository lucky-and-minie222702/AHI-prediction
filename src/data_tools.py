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
        if i == 11:
            continue
        seq_ECG = np.load(path.join("patients", f"patients_{i}_ECG.npy"))
        seq_SpO2 = np.load(path.join("patients", f"patients_{i}_SpO2.npy"))
        ann = np.load(path.join("patients", f"patients_{i}_anns.npy"))
        stage = np.load(path.join("patients", f"patients_{i}_stages.npy"))
        
        sequences_ECG.extend(seq_ECG.tolist())  # 30 seconds
        sequences_SpO2.extend(seq_SpO2.tolist()) 
        
        annotations.extend(ann.tolist())
        stages.extend(stage.tolist())

    sequences_ECG = np.array(sequences_ECG)
    sequences_SpO2 = np.array(sequences_SpO2)
    annotations = np.array(annotations)
    stages = np.array(stages)
    
    sequences_ECG = divide_signal(sequences_ECG, win_size=3000, step_size=500)  # 30s, step 5s
    sequences_SpO2 = divide_signal(sequences_SpO2, win_size=30, step_size=5)  # 30s, step 5s
    annotations = divide_signal(annotations, win_size=30, step_size=5)
    annotations = np.array([
        1 if np.count_nonzero(arr == 1) >= 10 else 0 for arr in annotations
    ])
    stages = divide_signal(stages, win_size=30, step_size=5)
    stages = np.round(np.mean(stages, axis=1))
    
    sequences_ECG = scaler.fit_transform(sequences_ECG.T).T  # scale
    
    # augment
    sequences_ECG = np.vstack(
        [sequences_ECG, sequences_ECG + np.random.normal(0.0, 0.05, sequences_ECG.shape), add_baseline_wander(sequences_ECG, frequency=0.05, amplitude=0.05, sampling_rate=100, flat_rate=0.5)]
    )
    sequences_SpO2 = np.concatenate(
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