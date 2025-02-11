import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import neurokit2 as nk
from scipy import signal
from itertools import groupby

def count_ones_zeros(binary_seq):
    groups = ["".join(g) for _, g in groupby(binary_seq)]
    count_ones = sum(1 for g in groups if g[0] == 1)
    count_zeros = sum(1 for g in groups if g[0] == 0)
    return count_ones, count_zeros

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

def add_baseline_wander(ecg_signal: np.ndarray, frequency: float, amplitude: float, sampling_rate: int, flat_rate: float, group_size: int = 300, num_groups: int = 8000):
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

def divide_signal(signal, win_size: int, step_size: int = None) -> np.ndarray:
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
        
    return np.array(segments)
  
def calc_cm(cm: np.ndarray | list):
    TP = cm[1][1] 
    FP = cm[0][1]  
    FN = cm[1][0]
    TN = cm[0][0]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Positive_accuracy = TP / (TP + FP)
    Negative_accuracy = TN / (TN + FN)

    return {
        "precision": precision, 
        "recall (sensivity)": recall, 
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
        
        res.append(list(stats.values()))
        
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

def calc_ecg(signals, splr: int, duration: int):
    """
    Return rpa, rri
    """
    rri_res = []
    rpa_res = []
    t = np.linspace(0, duration, splr * duration)
    for sig in signals:
        peaks = nk.ecg_findpeaks(sig, sampling_rate=splr, method="pantompkins1985")["ECG_R_Peaks"]  # https://www.researchgate.net/publication/375221357_Accelerated_Sample-Accurate_R-Peak_Detectors_Based_on_Visibility_Graphs

        if len(peaks) > 0:
            r_peaks_time = t[peaks]
            rpa = sig[peaks]
            rri = np.diff(r_peaks_time) 
        else:
            rpa = []
            rri = []
        
        max_rpa = max(max_rpa, len(rpa))
        max_rri = max(max_rri, len(rri))
        
        rpa_res.append(rpa)
        rri_res.append(rri)

    # print(max_rri, max_rpa)
    
    return rpa_res, rri_res

def calc_percentile(arr: np.ndarray, num: float) -> float:
    arr = sorted(arr)
    
    count = sum(1 for x in arr if x < num)
    
    percentile = (count / len(arr)) * 100
    return percentile

def fill_missing_with_mean(data):
    for i in range(len(data)):
        if data[i] is None:  # Detect missing values (None)
            # Find the starting and ending indices around the missing value
            start = i - 1
            while start >= 0 and data[start] is None:
                start -= 1
            
            end = i + 1
            while end < len(data) and data[end] is None:
                end += 1
            
            # Get the values at the start and end positions
            start_value = data[start] if start >= 0 else None
            end_value = data[end] if end < len(data) else None
            
            # Calculate the mean of the valid values
            surrounding_values = [val for val in [start_value, end_value] if val is not None]
            mean_value = np.mean(surrounding_values) if surrounding_values else 0
            
            # Replace the missing value with the calculated mean
            data[i] = mean_value
    
    return data

# Function to create a bandpass Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order: int = 4):
    nyquist = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order: int = 4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = signal.filtfilt(b, a, data)
    return y