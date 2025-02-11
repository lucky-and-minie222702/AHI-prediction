from numpy import ndim
from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

show_gpus()
no_logs()

info = open(path.join("data", "info.txt"), "r").readlines()
p_list = []
no_spo2 = []

for s in info:
    s = s[:-1:]
    if "*" in s:
        p_list.append(int(s[1::]))
        no_spo2.append(int(s[1::]))
    else:
        p_list.append(int(s))

p_list = [x for x in p_list if x not in no_spo2]
num_p = len(p_list)
        
        
def create_model():
    inp_rpa = layers.Input(shape=(None, 1))
    norm_inp_rpa = layers.Normalization()(inp_rpa)
    conv_rpa = layers.BatchNormalization()(conv_rpa)
    conv_rpa = layers.Activation("relu")(conv_rpa)
    conv_rpa = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(conv_rpa)
    conv_rpa = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv_rpa)
    conv_rpa = layers.BatchNormalization()(conv_rpa)
    conv_rpa = layers.Activation("relu")(conv_rpa)
    conv_rpa = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(conv_rpa)
    
    inp_rri = layers.Input(shape=(None, 1))
    norm_inp_rri = layers.Normalization()(inp_rri)
    conv_rri = layers.Conv1D(filters=32, kernel_size=3, padding="same")(norm_inp_rri)
    conv_rri = layers.BatchNormalization()(conv_rri)
    conv_rri = layers.Activation("relu")(conv_rri)
    conv_rri = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(conv_rri)
    conv_rri = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv_rri)
    conv_rri = layers.BatchNormalization()(conv_rri)
    conv_rri = layers.Activation("relu")(conv_rri)
    conv_rri = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(conv_rri)
    
    r_peak_features = layers.Concatenate(axis=-2)([conv_rpa, conv_rri])
    r_peak_features = layers.Conv1D(filters=128, kernel_size=3)(r_peak_features)
    r_peak_features = layers.BatchNormalization()(r_peak_features)
    r_peak_features = layers.Activation("relu")(r_peak_features)
    r_peak_features = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(r_peak_features)
    r_peak_features = layers.Conv1D(filters=256, kernel_size=3)(r_peak_features)
    r_peak_features = layers.BatchNormalization()(r_peak_features)
    r_peak_features = layers.Activation("relu")(r_peak_features)
    r_peak_features = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(r_peak_features)
    r_peak_features = SEBlock()(r_peak_features)
    
    inp = layers.Input(shape=(1000, 1))
    norm_inp = layers.Normalization()(inp)
    
    # down_sample
    ds_conv = layers.Conv1D(filters=64, kernel_size=11, strides=2, padding="same")(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(ds_conv)
    
    conv = ResNetBlock(1, ds_conv, 64, 9)
    conv = ResNetBlock(1, conv, 64, 7)
    conv = ResNetBlock(1, conv, 64, 5)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    
    conv = layers.Concatenate(axis=-2)([conv, r_peak_features])
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    
    # bottle-neck
    conv_bn = layers.Conv1D(filters=256, kernel_size=3, strides=2, padding="same")(conv)
    conv_bn = layers.BatchNormalization()(conv_bn)
    conv_bn = layers.Activation("relu")(conv_bn)
    
    rnn = layers.LSTM(128, return_sequences=True)(conv_bn)
    rnn = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(rnn)
    rnn = layers.LSTM(128, return_sequences=True)(rnn)
    
    # restore
    conv_r = layers.Conv1D(filters=256, kernel_size=3,padding="same")(rnn)
    conv_r = layers.BatchNormalization()(conv_r)
    conv_r = layers.Activation("relu")(conv_r)
    conv_r = layers.Conv1D(filters=512, kernel_size=3, padding="same")(conv_r)
    conv_r = layers.BatchNormalization()(conv_r)
    conv_r = layers.Add()([conv, conv_r])  # residual connection
    conv_r = layers.Activation("relu")(conv_r)
    
    se = SEBlock()(conv_r)
    
    fc = layers.GlobalAvgPool1D()(se)
    fc = layers.Dense(256)(fc)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Activation("relu")(fc)
    
    # preserved input shape
    pis = layers.Conv1D(filters=128, kernel_size=5, strides=5)(ds_conv)
    pis = layers.BatchNormalization()(pis)
    pis = layers.Activation("relu")(pis)
    pis = layers.Conv1D(filters=256, kernel_size=5, strides=5)(pis)
    pis = layers.BatchNormalization()(pis)
    pis = layers.Activation("relu")(pis)
    
    # full-segment output
    out = layers.Dot(axes=-1)([pis, fc])
    out = layers.Activation("sigmoid", name="full")(out)
    
    # single output
    out_s = layers.Dense(128)(fc)
    out_s = layers.BatchNormalization()(out_s)
    out_s = layers.Activation("relu")(out_s)
    out_s = layers.Dense(1, activation="sigmoid", name="single")(out_s)
    
    
    model = Model(inputs=[inp, inp_rpa, inp_rri], outputs=[out, out_s])
    model.compile(
        optimizer = "adam",
        loss = {
            "full": "binary_crossentropy",
            "single": "binary_crossentropy",
        },
        metrics = {
            "single": [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
        }
    )

    return model

model = create_model()
weights_path = path.join("weights", ".weights.h5")
model.save_weights(weights_path)

batch_size = 16
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 30,
    patience = 5,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
    monitor = "full_val_loss",
    mode = "min",
)

print("TRAINING\n")

# clean_method = ['pantompkins1985', 'hamilton2002', 'elgendi2010', 'vg']
# total_test_indices = []

for seg_len in range(10, 250, 10): # 10s -> 4m
    ecgs = []
    labels = []
    rpa = []
    rri = []
    
    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
        raw_label = raw_label[10:-10:]
    
        sig = divide_signal(raw_sig, win_size=(10+seg_len+10)*100, step_size=seg_len*100)
        label = divide_signal(raw_label, win_size=seg_len)

        ecgs.append(sig)
        labels.append(label)

    scaler = MinMaxScaler()
    
    ecgs = np.vstack(ecgs)
    ecgs = scaler.fit_transform(ecgs.T).T
    ecgs = np.array([nk.ecg.ecg_clean(e, sampling_rate=100, method="biosppy") for e in ecgs])
    rpa, rri = calc_ecg(ecgs, splr=100, duration=seg_len)
    full_labels = np.vstack(labels)
    single_labels = np.round(np.mean(full_labels, axis=-1))

    # print(np.unique(single_labels, return_counts=True))
    # exit()
    
    total_samples = len(ecgs)
    indices = np.arange(total_samples)
    indices = np.random.permutation(indices)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    test_size = total_samples - train_size - val_size
    
    print(f"Train - Test - Val: {train_size} - {test_size} - {val_size}")
    
    train_indices = indices[:train_size:]
    test_indices = indices[train_size:train_size+test_size:]
    val_indices = indices[train_size+test_size::]
    
    print(f"SEGMENT LENGTH: {seg_len}\n")
    model.load_weights(weights_path)
    model.fit(
        [ecgs[train_indices], rpa[train_indices], rri[train_indices]],
        [full_labels[train_indices], single_labels[train_indices]],
        epoch = 50,
        validation_data = (
            [ecgs[val_indices], rpa[val_indices], rri[val_indices]],
            [full_labels[val_indices], single_labels[val_indices]]
        ),
        batch_size = batch_size,
        callbacks = [cb_early_stopping, cb_checkpoint]
    )
    
    np.save(path.join("history", f"ecg_ah_test_{seg_len}"), train_indices)
    
    
# print("TESTING\n")
    
# for seg_len in range(10, 250, 10): # 10s -> 4m
#     ecgs = []
#     labels = []
#     rpa = []
#     rri = []
    
#     for p in p_list:
#         raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
#         raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
#         raw_label = raw_label[10:-10:]
    
#         sig = divide_signal(raw_sig, win_size=(10+seg_len+10)*100, step_size=seg_len*100)
#         label = divide_signal(raw_label, win_size=seg_len)

#         ecgs.append(sig)
#         labels.append(label)

#     scaler = MinMaxScaler()
    
#     ecgs = np.vstack(ecgs)
#     ecgs = scaler.fit_transform(ecgs.T).T
#     ecgs = np.array([nk.ecg.ecg_clean(e, sampling_rate=100, method="biosppy") for e in ecgs])
#     rpa, rri = calc_ecg(ecgs, splr=100, duration=seg_len)
#     full_labels = np.vstack(labels)
#     single_labels = np.round(np.mean(full_labels, axis=-1))

#     print(f"SEGMENT LENGTH: {seg_len}\n")