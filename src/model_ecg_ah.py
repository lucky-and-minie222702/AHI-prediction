from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

############

info = open(path.join("data", "info.txt"), "r").readlines()
raw_p_list = []
no_spo2 = []

for s in info:
    s = s[:-1:]
    if "*" in s:
        no_spo2.append(int(s[1::]))
    else:
        raw_p_list.append(int(s))

# p_list = [x for x in p_list if x not in no_spo2]
raw_p_list = np.array(raw_p_list)
num_p = len(raw_p_list)

############

show_gpus()

folds = 1
        
def create_model():
    #######
    # ECG #
    #######
    
    inp = layers.Input(shape=(97, 128))
    norm_inp = layers.Normalization()(inp)
    
    # down_sample
    ds_conv = layers.Conv1D(filters=32, kernel_size=7, strides=2, padding="same", kernel_regularizer=reg.l1_l2(l1=0.0001, l2=0.001))(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=2)(ds_conv)
    
    # deep
    conv = ResNetBlock(1, ds_conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3) 
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 1024, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 1024, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    se1 = SEBlock()(conv)
    
    out_s = layers.GlobalAvgPool1D()(se1)
    final_out_s = layers.Dense(1, activation="sigmoid")(out_s)

    
    model = Model(inputs=inp, outputs=final_out_s)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001), 
        loss = "binary_crossentropy",
        metrics = [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
    )

    return model

model = create_model()
show_params(model, "ecg_ah")
weights_path = path.join("history", "ecg_ah.weights.h5")
encoder = load_encoder()
model.save_weights(weights_path)

epochs = 400 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 256
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 350,
    patience = 10,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
)
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_ah"))
cb_lr = WarmupCosineDecayScheduler(warmup_epochs=20, total_epochs=400, target_lr=0.001, min_lr=1e-6)

seg_len = 30

ecgs = []
labels = []
rpa = []
rri = []
p_list = raw_p_list[:20:]

last_p = 0

for p in p_list:
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
    sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=1000)
    label = divide_signal(raw_label, win_size=(seg_len+1), step_size=10)
    
    if p == p_list[-2]:
        last_p = len(ecgs)

    ecgs.append(sig)
    labels.append(label)

scaler = MinMaxScaler()

val_ecgs = ecgs[last_p::]
val_labels = labels[last_p::]

ecgs = ecgs[:last_p:]
labels = labels[:last_p:]


# train
ecgs = np.vstack(ecgs)
ecgs = np.array([clean_ecg(e) for e in ecgs])
ecgs = np.vstack([
    ecgs,
    np.array([time_shift(e, shift_max=20) for e in ecgs]),
    np.array([bandpass(e, 100, 5, 35, 1) for e in ecgs]),
    np.array([bandpass(e, 100, 3, 45, 1) for e in ecgs]),
])
ecgs = scaler.fit_transform(ecgs.T).T

labels = np.vstack(labels)
labels = np.vstack([labels, labels, labels, labels, labels, labels])
mean_labels = np.mean(labels, axis=-1)
full_labels = np.round(mean_labels)
single_labels = np.array([l[15] for l in labels])


# val
val_ecgs = np.vstack(val_ecgs)
val_ecgs = np.array([clean_ecg(e) for e in val_ecgs])
# val_ecgs = np.vstack([
#     val_ecgs,
#     np.array([time_warp(e, sigma=0.2) for e in val_ecgs]),
#     np.array([time_shift(e, shift_max=20) for e in val_ecgs]),
#     np.array([bandpass(e, 100, 5, 35, 1) for e in val_ecgs]),
#     np.array([bandpass(e, 100, 3, 45, 1) for e in val_ecgs]),
#     np.array([frequency_noise(e, noise_std=0.15) for e in val_ecgs]),
# ])
val_ecgs = scaler.fit_transform(val_ecgs.T).T

val_labels = np.vstack(val_labels)
val_labels = np.vstack([val_labels, val_labels, val_labels, val_labels, val_labels, val_labels])
val_mean_labels = np.mean(val_labels, axis=-1)
val_full_labels = np.round(val_mean_labels)
val_single_labels = np.array([l[15] for l in val_labels])


# encode
ecgs = encoder.predict(ecgs, batch_size=256)
val_ecgs = encoder.predict(val_ecgs, batch_size=256)


print(f"Total samples: {len(labels)}\n")

total_samples = len(ecgs)

print(f"Train - Val: {len(ecgs)} - {len(val_ecgs)}")
class_counts = np.unique(val_single_labels, return_counts=True)[1]
print(f"Val: Class 0: {class_counts[0] // 6} - Class 1: {class_counts[1] // 6}")
class_counts = np.unique(single_labels, return_counts=True)[1]
print(f"Train: Class 0: {class_counts[0] // 6} - Class 1: {class_counts[1] // 6}\n")

sample_weights = [total_samples / class_counts[int(x)] for x in single_labels]
# sample_weights += mean_labels
sample_weights = np.array(sample_weights)

train_generator = DynamicAugmentedECGDataset(ecgs[:len(ecgs) // 6:], single_labels[:len(single_labels) // 6:],  ecgs, batch_size=256, num_augmented_versions=4, sample_weights=sample_weights).as_dataset()

steps_per_epoch = len(ecgs) // batch_size
steps_per_epoch //= 6

model.fit(
    train_generator,
    epochs = epochs,
    validation_data = (val_ecgs, val_single_labels),
    batch_size = batch_size,
    callbacks = [cb_early_stopping, cb_lr, cb_his],
    steps_per_epoch=steps_per_epoch,
)

res_file = open(path.join("history", "ah_res.txt"), "w")
res_file.close()

ecgs = []
labels = []
rpa = []
rri = []
# p_list = np.load(path.join("gen_data", f"fold_{i_fold}_test.npy"))
p_list = raw_p_list[20::]
print("\nTest result\n")

for p in p_list:
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])

    sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=100)
    label = divide_signal(raw_label, win_size=(seg_len+1), step_size=1)

    scaler = MinMaxScaler()

    ecgs = np.array(sig)
    ecgs = scaler.fit_transform(ecgs.T).T
    ecgs = encoder.predict(ecgs, batch_size=256)
    
    labels = np.array(label)
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    labels = np.array(label)
    mean_labels = np.mean(labels, axis=-1)
    full_labels = np.round(mean_labels)
    single_labels = np.array([l[15] for l in labels])

    class_counts = np.unique(single_labels, return_counts=True)[1]
    
    res_file = open(path.join("history", "ah_res.txt"), "a")
        
    print(f"\nBenh nhan {p}\n")
    print(f"\nBenh nhan {p}\n", file=res_file)
    
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n", file=res_file)
    
    raw_preds = model.predict(ecgs, batch_size=batch_size)
    single_preds = raw_preds
    single_preds = single_preds.flatten()

    np.save(path.join("history", f"ecg_ah_res_p{p}"), np.stack([single_labels, single_preds], axis=0))
    
    for t in np.linspace(0, 1, 11)[1:-1:]:
        t = round(t, 3)
        print(f"Threshold {t}: {acc_bin(single_labels, round_bin(single_preds, t))}")
        print(f"Threshold {t}: {acc_bin(single_labels, round_bin(single_preds, t))}", file=res_file)
    print()
    
    res_file.close()