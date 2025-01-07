import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import neurokit2 as nk

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s
    
def ith_int(s: list[str], i: int):
    count = 0
    for idx, val in enumerate(s):
        try:
            val = int(val)
            count += 1
        except:
            pass
        
        if count == i:
            return idx, val
        
def map_AHI(x):
    if x < 15:
        return 0
    elif 15 <= x <= 30:
        return 1
    else:
        return 2

def rearrange_array(arr: np.ndarray, group_size: int, num_groups: int) -> np.ndarray:
    # Convert input to numpy array
    arr = np.array(arr)
    
    # Count occurrences of 0s and 1s
    count_0 = np.sum(arr == 0)
    count_1 = np.sum(arr == 1)

    # Create alternating groups of specified size
    result = []
    for _ in range(num_groups):
        # Add group of 0s
        if count_0 > 0:
            size_0 = min(group_size, count_0)
            result.append(np.zeros(size_0, dtype=int))
            count_0 -= size_0
        # Add group of 1s
        if count_1 > 0:
            size_1 = min(group_size, count_1)
            result.append(np.ones(size_1, dtype=int))
            count_1 -= size_1

    if len(result) == 0:
        return np.array([])
    return np.concatenate(result)

def pad_arrays(arr1: np.ndarray, arr2: np.ndarray):
    len1, len2 = len(arr1), len(arr2)
    if len1 < len2:
        arr1 = np.pad(arr1, (0, len2 - len1), constant_values=0)
    return arr1, arr2

def add_baseline_wander(ecg_signal: np.ndarray, frequency: float, amplitude: float, sampling_rate: int, flat_rate: float, group_size: int = 500, num_groups: int = 5000):
    res = []
    og_size = ecg_signal.shape[1]
    p = ecg_signal.flatten()

    if num_groups == 0:
        flat_rate = 0.0
    
    flat_regions = np.random.choice([0, 1], size=len(p), p=[flat_rate, 1 - flat_rate])
    if num_groups != 0:
        flat_regions = rearrange_array(flat_regions, group_size, num_groups)
    t = np.arange(len(p)) / sampling_rate
    baseline = amplitude * np.sin(2 * np.pi * frequency * t)
    flat_regions, baseline = pad_arrays(flat_regions, baseline)
    baseline *= flat_regions
    
    res = p + baseline
    res = np.split(res, len(res) // og_size)
    return np.array(res)

def divide_signal(signals, win_size: int, step_size: int = None) -> np.ndarray:
    res = []
    for signal in signals:
        signal = np.array(signal)
        if step_size is None:
            # non-overlap
            step_size = win_size  
        
        num_segments = (len(signal) - win_size) // step_size + 1
        segments = []

        for i in range(0, num_segments * step_size, step_size):
            segment = signal[i:i + win_size]
            if len(segment) == win_size: 
                segments.append(segment)

        res.append(segments)
        
    return np.array(res)

def balancing_data(data: np.ndarray, majority_weight: float = 1.0) -> np.ndarray:  
    count0 = np.count_nonzero(data == 0)
    count1 = len(data) - count0
    count0 += count0 == count1  # avoid equal
    minority = [count0, count1].index(min(count0, count1))
    majority = [count0, count1].index(max(count0, count1))
    
    minority_data = np.where(data == minority)[0]
    majority_data = np.where(data == majority)[0]
    
    return np.random.permutation(np.concatenate([
        minority_data, majority_data[:int(len(minority_data) * majority_weight):]
    ]))
  
def calc_cm(cm: np.ndarray | list):
    TP = cm[1][1] 
    FP = cm[1][0]  
    FN = cm[0][1]
    TN = cm[0][0]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Positive_accuracy = TP / (TP + FP)
    Negative_accuracy = TN / (TN + FN)

    return {
        "precision": precision, 
        "recall": recall, 
        "sensitivity": sensitivity, 
        "specificity": specificity,
        "accuracy": accuracy,
        "positive_accuracy": Positive_accuracy,
        "negative_accuracy": Negative_accuracy,
    }
    
def calc_stats(data: np.ndarray | list[np.ndarray]):
    res = []
    for d in data:
        stats = {}
        stats["max"] = np.max(d)
        stats["min"] = np.min(d)
        stats["mean"] = np.mean(d)
        stats["median"] = np.median(d)
        stats["std"] = np.std(d)
        stats["var"] = np.var(d)
        stats["range"] = np.max(d) - stats["min"]
        
        res.append(stats)
        
    return res

def calc_time(start: str, end: str) -> int:
    # Parse times
    fmt = "%H:%M:%S"
    start_time = datetime.strptime(start, fmt)
    end_time = datetime.strptime(end, fmt)

    # handle overnight case
    if end_time < start_time:
        end_time += timedelta(days=1)

    elapsed_seconds = int((end_time - start_time).total_seconds())
    return elapsed_seconds

def calc_ecg(signals, fs: int = 100, duration: int = 30, max_rri: int = 60, max_rpa: int = 60):
    rri_res = []
    rpa_res = []
    t = np.linspace(0, duration, fs * duration)
    for sig in signals:
        peaks, _ = find_peaks(sig, height=0.5, distance=fs * 0.5)  # minimum 0.5s between beats <=> max 120 bpm

        r_peaks_time = t[peaks]
        rri = np.diff(r_peaks_time) 
        rpa = sig[peaks]
        
        max_rri = max(max_rri, len(rri))
        max_rpa = max(max_rpa, len(rpa))
        
        rri_res.append(rri)
        rpa_res.append(rpa)
    
    rri_res = np.array([np.pad(seq, (0, max_rri - len(seq)), 'constant', constant_values=0) for seq in rri_res])
    rpa_res = np.array([np.pad(seq, (0, max_rpa - len(seq)), 'constant', constant_values=0) for seq in rpa_res])
    
    return rpa_res, rri_res

def calc_percentile(arr: np.ndarray, num: float) -> float:
    arr = sorted(arr)
    
    count = sum(1 for x in arr if x < num)
    
    percentile = (count / len(arr)) * 100
    return percentile
