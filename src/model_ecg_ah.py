from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb

show_gpus()

input_scaler = StandardScaler()
        
# def create_model():
#     inp = layers.Input(shape=(249,))
#     norm_inp = layers.Normalization()(inp)
    
#     # feature selection
#     s = layers.Dense(249)(norm_inp)  # score
#     s = layers.BatchNormalization()(s)
#     s = layers.Activation("relu")(s)
#     s = layers.Dropout(rate=0.5)(s)
#     s = layers.Dense(249)(s)
#     s = layers.BatchNormalization()(s)
#     s = layers.Activation("sigmoid")(s)
#     fs = layers.Multiply()([s, norm_inp])
#     fs = layers.Normalization()(fs)
    
#     shortcut = layers.Dense(256)(fs)
#     shortcut = layers.BatchNormalization()(shortcut)
#     shortcut = layers.Activation("relu")(shortcut)
#     shortcut = layers.Dropout(rate=0.5)(shortcut)
#     shortcut = layers.Dense(256)(shortcut)
#     shortcut = layers.BatchNormalization()(shortcut)
#     shortcut = layers.Activation("relu")(shortcut)
#     shortcut = layers.Dropout(rate=0.5)(shortcut)
#     x = layers.Dense(512)(shortcut)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Dropout(rate=0.5)(x)
#     x = layers.Dense(512)(shortcut)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Dropout(rate=0.5)(x)
#     x = layers.Dense(256)(x)
#     x = layers.Add()([x, shortcut])
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Dropout(rate=0.5)(x)
#     x = layers.Dense(128)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     out = layers.Dense(1, activation="sigmoid")(x)

    # expanded_inp = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(fs)
    
    # conv = ResNetBlock(1, expanded_inp, 64, 3, change_sample=True, num_layers=3)
    # conv = ResNetBlock(1, conv, 64, 3, num_layers=3)
    # conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    # conv = ResNetBlock(1, conv, 128, 3, change_sample=True, num_layers=3)
    # conv = ResNetBlock(1, conv, 128, 3, num_layers=3)
    # conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    # conv = ResNetBlock(1, conv, 256, 3, change_sample=True, num_layers=3)
    # conv = ResNetBlock(1, conv, 256, 3, num_layers=3)
    # conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    # conv = ResNetBlock(1, conv, 512, 3, change_sample=True, num_layers=3)
    # conv = ResNetBlock(1, conv, 512, 3, num_layers=3)
    # conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    # fc = SEBlock(reduction_ratio=1)(conv)
    # fc = layers.GlobalAvgPool1D()(fc)
    # out = layers.Dense(1, activation="sigmoid")(fc)
    
    
    # model = Model(inputs=inp, outputs=out)
    # model.compile(
    #     optimizer = "adam", 
    #     loss = "binary_crossentropy",
    #     metrics = [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)] + ["binary_crossentropy"],
    # )

    # return model

# model = create_model()
# model.summary()
# show_params(model, "ecg_ah")
# weights_path = path.join("res", "ecg_ah.weights.h5")
# encoder = load_encoder()
# model.save_weights(weights_path)

# epochs = 1000 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

# batch_size = 512
# cb_early_stopping = cbk.EarlyStopping(
#     restore_best_weights = True,
#     start_from_epoch = 500,
#     patience = 10,
#     mode = "min",
#     monitor = "val_binary_crossentropy"
# )
# cb_checkpoint = cbk.ModelCheckpoint(
#     weights_path, 
#     save_best_only = True,
#     save_weights_only = True,
#     mode = "min",
#     monitor = "val_binary_crossentropy"
# )
# cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_ah"))
# cb_lr = WarmupCosineDecayScheduler(warmup_epochs=10, total_epochs=epochs, target_lr=0.001, min_lr=1e-6)
# cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-5)

params = {
    "objective": "binary",  # Binary classification
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",  # Gradient boosting decision tree
    "num_leaves": 96, 
    "learning_rate": 0.08,
    # "device_type": "cuda",
}

seg_len = 30
extra_seg_len = 10
step_size = 5

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
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=step_size*100)
    label = divide_signal(raw_label, win_size=seg_len, step_size=step_size)
    
    ecgs.append(sig)
    labels.append(label)

# train
	
test_ecgs = ecgs[25::]
test_labels = labels[25::]

ecgs = ecgs[:25:]
labels = labels[:25:]
 
 
ecgs = np.vstack(ecgs)
ecgs = np.vstack([
    ecgs,
    np.array([time_warp(e, sigma=0.08) for e in ecgs]),
    np.array([time_shift(e, shift_max=20) for e in ecgs]),
    np.array([add_noise(e, noise_std=0.005) for e in ecgs]),
])
ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in ecgs])

labels = np.vstack(labels)
labels = np.vstack([labels, labels, labels, labels])
labels = np.array([l[extra_seg_len:len(l)-extra_seg_len:] for l in labels])
mean_labels = np.mean(labels, axis=-1)
labels = np.round(mean_labels)

new_indices = downsample_indices_manual(labels)
# new_indices = np.arange(len(ecgs))
np.random.shuffle(new_indices)
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

psd = np.array([calc_psd(e, start_f=5, end_f=30) for e in ecgs])
psd = input_scaler.fit_transform(psd)
val_psd = np.array([calc_psd(e, start_f=5, end_f=30) for e in val_ecgs])
val_psd = input_scaler.transform(val_psd)
joblib.dump(input_scaler, path.join("res", "ecg_psd.scaler"))

# model.fit(
#     psd,
#     labels,
#     epochs = epochs,
#     validation_data = (val_psd, val_labels),
#     batch_size = batch_size,
#     callbacks = [cb_early_stopping, cb_lr, cb_his, cb_checkpoint],
# )
# model.load_weights(weights_path)

dtrain = lgb.Dataset(psd, label=labels)
dval = lgb.Dataset(val_psd, val_labels)
start_time = timer()
model = lgb.train(
    params, dtrain, 
    num_boost_round = 3000, 
    valid_sets=[dval], 
    valid_names=["Validation"], 
    callbacks = [lgb.early_stopping(stopping_rounds=20, first_metric_only=True)]
)
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")

model.save_model(path.join("res", "ecg_ah_lightgbm.txt"))
# model = lgb.Booster(model_file=path.join("res", "ecg_ah_lightgbm.txt"))
# input_scaler = joblib.load(path.join("res", "ecg_psd.scaler"))

print("\nTesting\n")

test_ecgs = np.vstack(test_ecgs)
test_ecgs = np.vstack([
    test_ecgs,
    np.array([time_warp(e, sigma=0.08) for e in test_ecgs]),
    np.array([time_shift(e, shift_max=20) for e in test_ecgs]),
    np.array([add_noise(e, noise_std=0.005) for e in test_ecgs]),
])
test_ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in test_ecgs])

test_labels = np.vstack(test_labels)
test_labels = np.vstack([test_labels, test_labels, test_labels, test_labels])
test_labels = np.array([l[extra_seg_len:len(l)-extra_seg_len:] for l in test_labels])
test_mean_labels = np.mean(test_labels, axis=-1)
test_labels = np.round(test_mean_labels)

new_indices = downsample_indices_manual(test_labels)
# new_indices = np.arange(len(ecgs))
np.random.shuffle(new_indices)
test_ecgs = test_ecgs[new_indices]
test_labels = test_labels[new_indices]

test_psd = np.array([calc_psd(e, start_f=5, end_f=30) for e in test_ecgs])
test_psd = input_scaler.transform(test_psd)

res_file = open(path.join("history", "ecg_ah_res.txt"), "w")

class_counts = np.unique(test_labels, return_counts=True)[1] 
print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")
print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n", file=res_file)

preds = model.predict(test_psd)

np.save(path.join("history", f"ecg_ah_predontest"), np.stack([test_labels, preds], axis=1))

print(f"AUC: {roc_auc_score(test_labels, preds)}")
print(f"Log loss: {log_loss(test_labels, preds)}")

print(f"AUC: {roc_auc_score(test_labels, preds)}", file=res_file)
print(f"Log loss: {log_loss(test_labels, preds)}", file=res_file)    


for t in np.linspace(0, 1, 11)[1:-1:]:
    t = round(t, 3)
    acc = acc_bin(test_labels, round_bin(preds, t))
    print(f"Threshold {t}: {acc}")
    print(f"Threshold {t}: {acc}", file=res_file)
print()

res_file.close()
    

if "additional_test" in sys.argv:
    pred = model.predict(psd)
    np.save(path.join("history", "ecg_ah_predontrain"), pred)
