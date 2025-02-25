from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler

input_scaler = MinMaxScaler()

show_gpus()

folds = 1
        
def create_model():
    inp = layers.Input(shape=(249, 1))
    
    # down-sample
    ds_conv = layers.Conv1D(filters=64, kernel_size=7, strides=2, padding="same")(inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(ds_conv)
    
    conv = ResNetBlock(1, ds_conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    fc = SEBlock(reduction_ratio=4)(conv)
    fc = layers.GlobalAvgPool1D()(fc)
    fc = layers.Dense(256)(fc)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Activation("relu")(fc)
    out = layers.Dense(1, activation="sigmoid")(fc)
    
    
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer = "adam", 
        loss = "binary_crossentropy",
        metrics = [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
    )

    return model

model = create_model()
# model.summary()
show_params(model, "ecg_ah")
weights_path = path.join("res", "ecg_ah.weights.h5")
# encoder = load_encoder()
# model.save_weights(weights_path)

epochs = 100 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 256
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 50,
    patience = 5,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
)
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_ah"))
cb_lr = WarmupCosineDecayScheduler(warmup_epochs=10, total_epochs=epochs, target_lr=0.001, min_lr=1e-5)
# cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-5)

seg_len = 30

ecgs = []
labels = []

test_ecgs = []
test_labels = []
p_list = good_p_list()

scaler = MinMaxScaler()

for idx, p in enumerate(p_list, start=1):
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:].flatten()
    
    sig = clean_ecg(raw_sig)    
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=1500)
    label = divide_signal(raw_label, win_size=seg_len, step_size=15)
    
    if idx >= 25:
        t_size = len(sig) // 2
        test_ecgs.append(sig[:t_size:])
        test_labels.append(label[:t_size:])
        sig = sig[t_size::]
        label = label[t_size::]

    ecgs.append(sig)
    labels.append(label)

# train
ecgs = np.vstack(ecgs)
np.random.shuffle(ecgs)
ecgs = np.vstack([
    ecgs,
    np.array([time_warp(e, sigma=0.08) for e in ecgs]),
    np.array([time_shift(e, shift_max=20) for e in ecgs]),
    np.array([add_noise(e, noise_std=0.005) for e in ecgs]),
])
ecgs = scaler.fit_transform(ecgs.T).T
num_augment = 3

labels = np.vstack(labels)
labels = np.vstack([labels, labels, labels, labels])
mean_labels = np.mean(labels, axis=-1)
labels = np.round(mean_labels)

new_indices = downsample_indices_manual(labels)
ecgs = ecgs[new_indices]
labels = labels[new_indices]

print(f"Total samples: {len(labels)}\n")
total_samples = len(ecgs)
indices = np.arange(total_samples)
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(22022009))

val_ecgs = ecgs[val_indices]
val_labels = labels[val_indices]
ecgs = ecgs[train_indices]
labels = labels[train_indices]

print(f"Train - Val: {len(ecgs)} - {len(val_ecgs)}")
class_counts = np.unique(val_labels, return_counts=True)[1] 
print(f"Val: Class 0: {class_counts[0]} - Class 1: {class_counts[1]}")
class_counts = np.unique(labels, return_counts=True)[1]
print(f"Train: Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")

sample_weights = [total_samples / class_counts[int(x)] for x in labels]
sample_weights += mean_labels[train_indices]
sample_weights = np.array(sample_weights)

psd = np.array([calc_psd(e, start_f=5, end_f=30) for e in ecgs])
psd = input_scaler.fit_transform(psd)
val_psd = np.array([calc_psd(e, start_f=5, end_f=30) for e in val_ecgs])
val_psd  =input_scaler.transform(val_psd)
joblib.dump(input_scaler, path.join("res", "ecg_psd.scaler"))

model.fit(
    psd,
    labels,
    epochs = epochs,
    validation_data = (val_psd, val_labels),
    batch_size = batch_size,
    callbacks = [cb_early_stopping, cb_lr, cb_his, cb_checkpoint],
)
model.load_weights(weights_path)
input_scaler  =joblib.load(path.join("res", "ecg_psd.scaler"))

print("\nTesting\n")

res_file = open(path.join("history", "ecg_ah_res.txt"), "w")
res_file.close()

# test
test_psd = [
    np.vstack([calc_psd(e, start_f=5, end_f=30) for e in p_ecg]) for p_ecg in test_ecgs
]
test_labels = [
    np.mean(l, axis=-1) for l in test_labels
]
mean_res = [[] for _ in range(9)]

for idx, p in enumerate(good_p_list()[25::]):
    res_file = open(path.join("history", "ah_res.txt"), "a")
    
    print(f"\nBenh nhan {p}\n")
    print(f"\nBenh nhan {p}\n", file=res_file)
    
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n", file=res_file)
    
    preds = model.predict(input_scaler.transform(test_psd[idx]), batch_size=batch_size).flatten()
    print(preds.shape, test_labels[idx].shape)
    
    np.save(path.join("history", f"ecg_ah_res_p{p}"), np.stack([test_labels[idx], preds], axis=0))
    
    for t in np.linspace(0, 1, 11)[1:-1:]:
        t = round(t, 3)
        acc = acc_bin(test_labels[idx], round_bin(preds, t))
        print(f"Threshold {t}: {acc}")
        mean_res[int(t*10) - 1].append(acc)
        print(f"Threshold {t}: {acc}", file=res_file)
    print()
    
    res_file.close()
    
res_file = open(path.join("history", "ah_res.txt"), "a")

print("\nMean Accuracy\n")
print("\nMean Accuracy\n", file=res_file)

mean_res = np.array(mean_res)
mean_res = np.mean(mean_res, axis=-1)

for idx, acc in enumerate(mean_res, start=1):
    print(f"Threshold 0.{idx}: {acc}")
    print(f"Threshold 0.{idx}: {acc}", file=res_file)

res_file.close()