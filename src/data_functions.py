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
        res.append(t + baseline)
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