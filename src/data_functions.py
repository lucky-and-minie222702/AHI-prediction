import numpy as np

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

def add_baseline_wander(ecg_signal, frequency: float, amplitude: float, sampling_rate: int):
    res = []
    for p in ecg_signal:
        t = np.arange(len(p)) / sampling_rate
        baseline = amplitude * np.sin(2 * np.pi * frequency * t)
        res.append(p + baseline)
    return np.array(res)

def divide_signal(signals, win_size, step_size=None):
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
    
    sensivity = TP / (TP + FN)
    specifity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Positive_accuracy = TP / (TP + FP)
    Negative_accuracy = TN / (TN + FN)

    return {
        "precision": precision, 
        "recall": recall, 
        "sensivity": sensivity, 
        "specifity": specifity,
        "accuracy": accuracy,
        "positive_accuracy": Positive_accuracy,
        "negative_accuracy": Negative_accuracy,
    }
    
def calc_stats(data: np.ndarray | list[np.ndarray]):
    res = []
    for d in data:
        stats = {}
        # stats["max"] = np.max(d)
        stats["min"] = np.min(d)
        stats["mean"] = np.mean(d)
        # stats["median"] = np.median(d)
        stats["std"] = np.std(d)
        stats["var"] = np.var(d)
        # stats["range"] = stats["max"] - stats["min"]
        
        res.append(stats)
        
    return res