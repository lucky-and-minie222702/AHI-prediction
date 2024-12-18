from data_functions import *
import sys
from os import path
from sklearn.model_selection import train_test_split

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
        sequences_SpO2 += np.split(seq_SpO2, len(seq_SpO2) // 30)
        
        annotations.extend(ann.tolist())
        stages.extend(stage.tolist())

    sequences_ECG = np.array(sequences_ECG)
    sequences_SpO2 = np.array(seq_SpO2)
    annotations = np.array(annotations)
    stages = np.array(stages)
    
    sequences_ECG = np.vstack(
        [sequences_ECG, sequences_ECG + np.random.normal(0, 0.003, sequences_ECG.shape), add_baseline_wander(sequences_ECG, frequency=0.05, amplitude=0.05, sampling_rate=100)]
    )
    sequences_SpO2 = np.vstack(
        [sequences_SpO2, sequences_SpO2 + np.random.normal(0, 0.01, sequences_SpO2.shape), sequences_SpO2 + np.random.normal(0, 0.015, sequences_SpO2.shape)]
    )
    annotations = np.concatenate(
        [annotations, annotations, annotations]
    )
    stages = np.concatenate(
        [stages, stages, stages]
    )
    
    indices = np.arange(len(annotations))
    train_indices, test_indices = train_test_split(indices, test_size=0.2,random_state=np.random.randint(69696969))
    
    np.save(path.join("patients", "merged_ECG"), sequences_ECG)
    np.save(path.join("patients", "merged_SpO2"), sequences_SpO2)
    np.save(path.join("patients", "merged_anns"), annotations)
    np.save(path.join("patients", "merged_stages"), stages)
    np.save(path.join("patients", "train_indices"), train_indices)
    np.save(path.join("patients", "test_indices"), test_indices)