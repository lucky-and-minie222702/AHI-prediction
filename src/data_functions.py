import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import neurokit2 as nk
from scipy import signal
from itertools import groupby
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, log_loss
from typing import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error, root_mean_squared_error
import sys
import scipy.fftpack
from sklearn.utils import resample

def count_ones_zeros(binary_seq):
    groups = ["".join(g) for _, g in groupby(binary_seq)]
    count_ones = sum(1 for g in groups if g[0] == 1)
    count_zeros = sum(1 for g in groups if g[0] == 0)
    return count_ones, count_zeros

def count_groups(binary_list):
    if len(binary_list) == 0:
        return []
    
    groups = {0: [], 1: []}
    start = 0
    current_value = binary_list[0]

    for i in range(1, len(binary_list)):
        if binary_list[i] != current_value:
            # start index, count
            groups[current_value].append((start, i - start))
            start = i
            current_value = binary_list[i]
    
    groups[current_value].append((start, len(binary_list) - start))

    return groups

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
    
    max_rpa = 0
    max_rri = 0
    t = np.linspace(0, duration, splr * duration)
    for sig in signals:
        peaks = nk.ecg_findpeaks(sig, sampling_rate=splr, method="pantompkins1985")["ECG_R_Peaks"]

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
    
    rri_res = np.array([np.pad(seq, (0, max_rri - len(seq)), 'constant', constant_values=0) for seq in rri_res])
    rpa_res = np.array([np.pad(seq, (0, max_rpa - len(seq)), 'constant', constant_values=0) for seq in rpa_res])
    
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

def time_warp(ecg_signal, sigma: float):
    orig_time = np.arange(len(ecg_signal))

    time_warping = np.cumsum(np.random.normal(0, sigma, size=len(ecg_signal)))

    warped_time = orig_time + time_warping
    warped_time = np.clip(warped_time, 0, len(ecg_signal) - 1)

    warped_time, unique_indices = np.unique(warped_time, return_index=True)
    ecg_signal_unique = ecg_signal[unique_indices]

    interp_func = interp1d(warped_time, ecg_signal_unique, kind="cubic", fill_value="extrapolate")
    warped_ecg = interp_func(orig_time)

    return warped_ecg

def equally_dis(num_classes: int) -> List[int]:
    ele = 1.0 / num_classes
    last_ele = 1 - ele * (num_classes - 1)
    return [ele for _ in range(num_classes - 1)] + [last_ele]
    
def split_classes(y: np.ndarray, class_ratio: List[float] = None, max_total_samples: int = None):
    class_counts = np.unique(y, return_counts=True)[1]
    num_classes = len(class_counts)
    if class_ratio is None:
        class_ratio = equally_dis(num_classes)
    
    # bin search
    l = 0
    r = len(y)
    m = 0
    while l <= r:
        m = (l+r)//2
        wrong = any([(m * class_ratio[i]) > class_counts[i] for i in range(num_classes)])
        if wrong:
            r = m -1
        else:
            l = m + 1
    
    total_samp = m
    if max_total_samples is not None:
        total_samp = min(total_samp, max_total_samples)

    class_idx = []
    for cls, i in enumerate(class_ratio):
        k = int(total_samp * i)
        class_idx.extend(np.random.choice(np.where(y==cls)[0], k, replace=False))
        
    return np.array(class_idx), np.array([int(total_samp * r) for r in class_ratio])

def print_confusion_matrix(cm: np.ndarray | List[List[int]], labels = None):
    if labels is None:
        labels = list(map(str, range(len(cm))))
    assert max([len(l) for l in labels]) <= 6, "Labels length for confusion matrix must not exceed 6"
    print("Confusion Matrix:")
    print(" " * 10, "Predicted", sep="")
    print(" " * 10, " ".join(f"{label:>6}" for label in labels), sep="")
    print(" " * 10 + "-" * (7 * (len(labels))), sep="")
    remain_label = "Actual"
    for i, row in enumerate(cm):
        print(f"{remain_label[i] if i < len(remain_label) else ' '} {labels[i]:>6} |", " ".join(f"{val:>6}" for val in row), " |", sep="")
    print(remain_label[len(cm)] if len(cm) < len(remain_label) else " ", " " * 9, "-" * (7 * (len(labels))), sep="")
    for s in remain_label[len(cm)+1::]:
        print(s)

def show_res(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm, labels=labels)
    print("\nClassification Report:")
    if cm.shape[-1] == 2:  # 2-class
        tn, fp, fn, tp = cm.ravel()
        # Compute classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred,zero_division=0)  # Sensitivity
        specificity = tn / (tn + fp)  # True Negative Rate
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)  # Matthews Correlation Coefficient
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        # Print evaluation metrics
        print("Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

def  count_first_ele(lst):
    ele = lst[0]
    res = 0
    for i in range(len(lst)):
        if lst[i] == ele:
            res += 1
        else:
            break
    return res

def show_res_regression(y_true, y_pred):
    # Number of data points and predictors (for Adjusted R²)
    n = len(y_true)
    p = 1  # Modify this if using multiple features in a regression model

    # Compute regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Adjusted R²
    mape = mean_absolute_percentage_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)

    # Print evaluation metrics
    print("Regression Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R² Score: {adj_r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Median Absolute Error: {median_ae:.4f}")
    
def bandpass(signal_noisy, fs, low_cutoff_hz, high_cutoff_hz, order):
    normalized_low_cutoff = low_cutoff_hz / (fs / 2) 
    normalized_high_cutoff = high_cutoff_hz / (fs / 2)

    b_bp, a_bp = signal.butter(order, [normalized_low_cutoff, normalized_high_cutoff], btype='band')

    return signal.filtfilt(b_bp, a_bp, signal_noisy)

def clean_ecg(sig):
    return bandpass(sig, 100, 5, 30, 2)


class Tee:
    def reset():
        sys.stdout.file.close() 
        sys.stdout = sys.__stdout__
    
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
        
def calc_psd(sig, start_f, end_f):
    f, Pxx = signal.welch(sig, fs=100, nperseg=1000)
    start = None
    end = None
    for i in range(len(f)):
        if f[i] > start_f and start is None:
            start = i

        if f[i] > end_f and end is None:
            end = i - 1
            break

    f = f[start:end:]
    Pxx = Pxx[start:end:]
    
    return np.array(Pxx)

def calc_fft(sig):
    return np.fft.fft(sig).real[1::]

def time_shift(ecg_signal, shift_max):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(ecg_signal, shift)

def frequency_noise(ecg_signal, noise_std):
    fft_ecg = scipy.fftpack.fft(ecg_signal)
    noise = np.random.normal(0, noise_std, size=fft_ecg.shape)
    return np.real(scipy.fftpack.ifft(fft_ecg + noise))

def add_noise(sig, noise_std):
    return sig + np.random.normal(0, noise_std, size=sig.shape)

def round_bin(arr, threshold=0.5):
    return (arr >= threshold).astype(int)

def acc_bin(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Arrays must have the same shape"
    return np.mean(y_true == y_pred)

def good_p_list():
    num_p = 37
    bad_list = [20, 34]
    ans = [x for x in range(1, num_p+1) if x not in bad_list]
    return ans

def split_classes(y: np.ndarray, class_ratio: List[float] = None, max_total_samples: int = None):
    class_counts = np.unique(y, return_counts=True)[1]
    num_classes = len(class_counts)
    if class_ratio is None:
        class_ratio = equally_dis(num_classes)
    
    # bin search
    l = 0
    r = len(y)
    m = 0
    while l <= r:
        m = (l+r)//2
        wrong = any([(m * class_ratio[i]) > class_counts[i] for i in range(num_classes)])
        if wrong:
            r = m -1
        else:
            l = m + 1
    
    total_samp = m
    if max_total_samples is not None:
        total_samp = min(total_samp, max_total_samples)

    class_idx = []
    for cls, i in enumerate(class_ratio):
        k = int(total_samp * i)
        class_idx.extend(np.random.choice(np.where(y==cls)[0], k, replace=False))
        
    return np.array(class_idx), np.array([int(total_samp * r) for r in class_ratio])

def downsample_indices_manual(y):
    y = np.array(y)  # Ensure it's a NumPy array
    major_class = max(set(y), key=list(y).count)  # Find majority class
    
    # Get indices of major and minor classes
    majority_indices = np.where(y == major_class)[0]
    minority_indices = np.where(y != major_class)[0]
    
    # Downsample majority class
    downsampled_majority = resample(majority_indices, 
                                    replace=False, 
                                    n_samples=len(minority_indices), 
                                    random_state=42)
    
    # Combine indices
    downsampled_indices = np.concatenate([downsampled_majority, minority_indices])
    return downsampled_indices

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)