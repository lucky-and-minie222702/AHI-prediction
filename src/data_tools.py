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

    # keep patient 25 for visualization
    for i in range(1, 25):
        seq_ECG = np.load(path.join("patients", f"patients_{i}_ECG.npy"))
        seq_SpO2 = np.load(path.join("patients", f"patients_{i}_SpO2.npy"))
        ann = np.load(path.join("patients", f"patients_{i}_anns.npy"))
        stage = np.load(path.join("patients", f"patients_{i}_stages.npy"))
        
        sequences_ECG += np.split(seq_ECG, len(seq_ECG) // 1000)
        sequences_SpO2 += np.split(seq_SpO2, len(seq_SpO2) // 10)
        
        annotations.extend(ann.tolist())
        stages.extend(stage.tolist())

    sequences_ECG = np.array(sequences_ECG)
    sequences_SpO2 = np.array(sequences_SpO2)
    annotations = np.array(annotations)
    stages = np.array(stages)

    sequences_ECG = scaler.fit_transform(sequences_ECG.T).T

    sequences_ECG = np.vstack(
        [sequences_ECG, add_baseline_wander(sequences_ECG, frequency=0.1, amplitude=0.1, sampling_rate=100), sequences_ECG + np.random.normal(0.0, 0.05, sequences_ECG.shape)]
    )
    sequences_SpO2 = np.vstack(
        [sequences_SpO2, sequences_SpO2 + np.random.normal(0.0, 0.01, sequences_SpO2.shape)]
    )
    
    _, counts = np.unique(annotations, return_counts=True)
    print(f"Annotations: [0]: {counts[0]} | [1]: {counts[1]}")
    _, counts = np.unique(stages, return_counts=True)
    print(f"Stages: [0]: {counts[0]} | [1]: {counts[1]}")

    np.save(path.join("patients", "merged_ECG"), sequences_ECG)
    np.save(path.join("patients", "merged_SpO2"), sequences_SpO2)
    np.save(path.join("patients", "merged_anns"), annotations)
    np.save(path.join("patients", "merged_stages"), stages)