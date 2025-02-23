from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler

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
    # rpa #
    #######
    rpa_inp = layers.Input(shape=(None, 1))
    norm_inp = layers.Normalization()(rpa_inp)
    
    # down_sample
    rpa_ds_conv = layers.Conv1D(filters=32, kernel_size=3, padding="same")(norm_inp)
    rpa_ds_conv = layers.BatchNormalization()(rpa_ds_conv)
    rpa_ds_conv = layers.Activation("relu")(rpa_ds_conv)
    
    rpa_conv = ResNetBlock(1, rpa_ds_conv, 64, 3, change_sample=True)
    rpa_conv = ResNetBlock(1, rpa_conv, 64, 3)
    
    rpa_conv = layers.SpatialDropout1D(rate=0.1)(rpa_conv)
    
    rpa_conv = ResNetBlock(1, rpa_conv, 128, 3, change_sample=True)
    rpa_conv = ResNetBlock(1, rpa_conv, 128, 3, )
    
    rpa_conv = layers.SpatialDropout1D(rate=0.1)(rpa_conv)


    #######
    # rri #
    #######
    rri_inp = layers.Input(shape=(None, 1))
    norm_inp = layers.Normalization()(rri_inp)
    
    # down_sample
    rri_ds_conv = layers.Conv1D(filters=32, kernel_size=3, padding="same")(norm_inp)
    rri_ds_conv = layers.BatchNormalization()(rri_ds_conv)
    rri_ds_conv = layers.Activation("relu")(rri_ds_conv)
    
    rri_conv = ResNetBlock(1, rri_ds_conv, 64, 3, change_sample=True)
    rri_conv = ResNetBlock(1, rri_conv, 64, 3)
    
    rri_conv = layers.SpatialDropout1D(rate=0.1)(rri_conv)
    
    rri_conv = ResNetBlock(1, rri_conv, 128, 3, change_sample=True)
    rri_conv = ResNetBlock(1, rri_conv, 128, 3, )
    
    rri_conv = layers.SpatialDropout1D(rate=0.1)(rri_conv)
    
    # bottle-neck attention merge
    rpa_btn_conv = layers.Conv1D(filters=64, kernel_size=3, padding="same")(rpa_conv)
    rpa_btn_conv = layers.BatchNormalization()(rpa_btn_conv)
    rpa_btn_conv = layers.Activation("relu")(rpa_btn_conv)
    
    rri_btn_conv = layers.Conv1D(filters=64, kernel_size=3, padding="same")(rri_conv)
    rri_btn_conv = layers.BatchNormalization()(rri_btn_conv)
    rri_btn_conv = layers.Activation("relu")(rri_btn_conv)
    
    att_merge = MyAtt(depth=32, num_heads=8, dropout_rate=0.1)(rpa_btn_conv, rri_btn_conv, rri_btn_conv)
    
    
    # final conv
    f_conv = ResNetBlock(1, att_merge, 512, 3, change_sample=True)
    f_conv = ResNetBlock(1, f_conv, 512, 3)
    f_conv = layers.SpatialDropout1D(rate=0.1)(f_conv)
    
    # fully-connected
    se = SEBlock()(f_conv)
    f_out = layers.GlobalAvgPool1D()(se)
    f_out = layers.Dropout(rate=0.1)(f_out)
    f_out = layers.Dense(1, activation="sigmoid")(f_out)

    
    model = Model(inputs=[rpa_inp, rri_inp], outputs=f_out)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001), 
        loss = "binary_crossentropy",
        metrics = [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
    )

    return model

model = create_model()
show_params(model, "ecg_ah")
weights_path = path.join("history", "ecg_ah.weights.h5")
# encoder = load_encoder()
# model.save_weights(weights_path)

epochs = 300 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 256
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 100,
    patience = 10,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
)
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_ah"))
cb_lr = WarmupCosineDecayScheduler(warmup_epochs=10, total_epochs=epochs, target_lr=0.001, min_lr=1e-6)

seg_len = 30

ecgs = []
labels = []
rpa = []
rri = []
p_list = raw_p_list[:20:]

last_p = 0

scaler = MinMaxScaler()

for p in p_list:
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
    
    sig = clean_ecg(raw_sig)
    sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=1000)
    label = divide_signal(raw_label, win_size=(seg_len+1), step_size=10)
    
    if p == p_list[-2]:
        last_p = len(ecgs)

    ecgs.append(sig)
    labels.append(label)

val_ecgs = ecgs[last_p::]
val_labels = labels[last_p::]

ecgs = ecgs[:last_p:]
labels = labels[:last_p:]


# train
ecgs = np.vstack(ecgs)
ecgs = np.vstack([
    ecgs,
    np.array([time_warp(e, sigma=0.2) for e in ecgs]),
    np.array([time_shift(e, shift_max=20) for e in ecgs]),
    np.array([frequency_noise(e, noise_std=0.15) for e in ecgs]),
    np.array([add_noise(e, noise_std=0.005) for e in ecgs]),
])
ecgs = scaler.fit_transform(ecgs.T).T
rpa, rri = calc_ecg(ecgs, 100, seg_len + 1)

labels = np.vstack(labels)
labels = np.vstack([labels, labels, labels, labels, labels])
mean_labels = np.mean(labels, axis=-1)
full_labels = np.round(mean_labels)
single_labels = np.array([l[15] for l in labels])
# single_labels  =np.expand_dims(single_labels, axis=-1)


# val
val_ecgs = np.vstack(val_ecgs)
val_ecgs = np.vstack([
    val_ecgs,
    np.array([time_warp(e, sigma=0.2) for e in val_ecgs]),
    np.array([time_shift(e, shift_max=20) for e in val_ecgs]),
    np.array([frequency_noise(e, noise_std=0.15) for e in val_ecgs]),
    np.array([add_noise(e, noise_std=0.005) for e in val_ecgs]),
])
val_ecgs = scaler.fit_transform(val_ecgs.T).T
val_rpa, val_rri = calc_ecg(val_ecgs, 100, seg_len + 1)

val_labels = np.vstack(val_labels)
val_labels = np.vstack([val_labels, val_labels, val_labels, val_labels, val_labels])
val_mean_labels = np.mean(val_labels, axis=-1)
val_full_labels = np.round(val_mean_labels)
val_single_labels = np.array([l[15] for l in val_labels])


num_augment = 5


print(f"Total samples: {len(labels)}\n")

total_samples = len(ecgs)

print(f"Train - Val: {len(ecgs)} - {len(val_ecgs)}")
class_counts = np.unique(val_single_labels, return_counts=True)[1]
print(f"Val: Class 0: {class_counts[0]} - Class 1: {class_counts[1]}")
class_counts = np.unique(single_labels, return_counts=True)[1]
print(f"Train: Class 0: {class_counts[0] // num_augment} - Class 1: {class_counts[1] // num_augment}\n")

sample_weights = [total_samples / class_counts[int(x)] for x in single_labels]
# sample_weights += mean_labels
sample_weights = np.array(sample_weights)

# train_generator = DynamicAugmentedECGDataset([rpa[:len(rpa) // num_augment:], rri[:len(rri) // num_augment:]], [single_labels[:len(single_labels) // num_augment:]],  [rpa, rri], [single_labels], batch_size=batch_size, num_augmented_versions=num_augment, sample_weights=sample_weights).as_dataset()

# steps_per_epoch = len(ecgs) // batch_size
# steps_per_epoch //= num_augment

model.fit(
    [rpa, rri],
    single_labels,
    epochs = epochs,
    validation_data = ([val_rpa, val_rri], val_single_labels),
    batch_size = batch_size,
    callbacks = [cb_early_stopping, cb_lr, cb_his, cb_checkpoint],
    # steps_per_epoch = steps_per_epoch,
)

model.load_weights(weights_path)

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

    sig = clean_ecg(raw_sig)
    sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=100)
    label = divide_signal(raw_label, win_size=(seg_len+1), step_size=1)

    scaler = MinMaxScaler()

    ecgs = np.array(sig)
    ecgs = scaler.fit_transform(ecgs.T).T
    
    rpa, rri = calc_ecg(ecgs, 100, seg_len + 1)
    
    labels = np.array(label)
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
    
    raw_preds = model.predict([rpa, rri], batch_size=batch_size)
    single_preds = raw_preds
    single_preds = single_preds.flatten()

    np.save(path.join("history", f"ecg_ah_res_p{p}"), np.stack([single_labels, single_preds], axis=0))
    
    for t in np.linspace(0, 1, 11)[1:-1:]:
        t = round(t, 3)
        print(f"Threshold {t}: {acc_bin(single_labels, round_bin(single_preds, t))}")
        print(f"Threshold {t}: {acc_bin(single_labels, round_bin(single_preds, t))}", file=res_file)
    print()
    
    res_file.close()
