from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb

show_gpus()

seg_len = 10
extra_seg_len = 0
step_size = 10

memory_ecgs = []
memory_labels = []

test_ecgs = []
test_labels = []

p_list = good_p_list()

scaler = StandardScaler()
last_p = 0

for idx, p in enumerate(p_list, start=1):
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:].flatten()
    
    sig = clean_ecg(raw_sig)    
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=step_size*100)
    label = divide_signal(raw_label, win_size=seg_len, step_size=step_size)
    
    memory_ecgs.append(sig)
    memory_labels.append(label) 
    
    if idx == 10:
        last_p  = sum([len(e) for e in memory_ecgs])

memory_ecgs = np.vstack(memory_ecgs)
memory_ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in memory_ecgs])

memory_labels = np.vstack(memory_labels)
memory_labels = np.array([l[extra_seg_len:len(l)-extra_seg_len:] for l in memory_labels])
memory_labels = np.mean(memory_labels, axis=-1)
memory_labels = np.round(memory_labels)

test_ecgs = memory_ecgs[last_p::]
test_labels = memory_labels[last_p::]
memory_ecgs = memory_ecgs[:last_p:]
memory_labels = memory_labels[:last_p:]

memory_indices = np.arange(len(memory_labels))
test_indices = np.arange(len(test_labels))
np.random.shuffle(memory_indices)
np.random.shuffle(test_indices)

memory_ecgs = memory_ecgs[memory_indices]
memory_labels = memory_labels[memory_indices]
test_ecgs = test_ecgs[test_indices]
test_labels = test_labels[test_indices]

test_preds = []
print("Test - Prediction")
for idx in range(len(test_labels)):
    test_preds.append(np.argmax(predict_using_ecg_encoder(memory_ecgs, memory_labels, test_ecgs[idx], num_sample_per_class=5000)))
    print(f"{test_labels[idx] - test_preds[idx]}", end="\r")
    sys.stdout.flush()

print()
test_preds = np.array(test_preds)
print("Accuracy", sum(test_labels == test_preds) / len(test_labels))
np.save(path.join("history", "ah_test_res"), np.stack([test_labels,  test_preds], axis=1))    