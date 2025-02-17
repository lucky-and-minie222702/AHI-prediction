from numpy import real
from data_functions import *
from os import path

num_p = 30
res = []

for p in range(1, num_p+1):
    raw_ah = np.load(path.join("history", f"ecg_ah_res_p{p}.npy"))
    real_ah = raw_ah[0]
    ai_ah = raw_ah[1]
    
    print("\nAPNEA HYPOPNEA EVENTS\n")
    print("Metrics\n")
    show_res(real_ah, ai_ah)
    print("\n")
    real_ah_groups = count_groups(real_ah)
    ai_ah_groups = count_groups(ai_ah)
    print("MAE of duration:", mean_absolute_error([x[1] for x in real_ah_groups], [x[1] for x in ai_ah_groups]))
    print("MAE of start second:", mean_absolute_error([x[0] for x in real_ah_groups], [x[0] for x in ai_ah_groups]))
    print("MAE of end second:", mean_absolute_error([x[0]+x[1]-1 for x in real_ah_groups], [x[0]+x[1]-1 for x in ai_ah_groups]))
    
    raw_wake = np.load(path.join("history", f"ecg_stage_res_p{p}.npy"))
    real_wake = raw_wake[0]
    ai_wake = raw_wake[1]
    
    print("\nSLEEP / WAKES EVENTS\n")
    show_res(real_wake, ai_wake)
    
    real_ah_count = sum([1 for x in real_ah_groups if x[2] == 1])
    ai_ah_count = sum([1 for x in ai_ah_groups if x[2] == 1])
    
    real_sleep_time = (len(real_wake) - sum(real_wake)) * 30
    ai_sleep_time = (len(ai_wake) - sum(ai_wake)) * 30
    
    real_ahi = real_ah_count / (real_sleep_time / 60 / 60)
    ai_ahi = ai_ah_count / (ai_sleep_time / 60 / 60)
    
    real_ahi = round(real_ahi, 2)
    ai_ahi = round(ai_ahi, 2)
    
    print(f"Benh nhan {p}:")
    print(real_ahi, ai_ahi)
    res.append([real_ahi, ai_ahi])
    
np.save("final_result", np.array(res))