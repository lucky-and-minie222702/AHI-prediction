from numpy import ndim
from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

show_gpus()

folds = 2
        
def create_model():
    inp_rpa = layers.Input(shape=(None, 1))
    norm_inp_rpa = layers.Normalization()(inp_rpa)
    conv_rpa = ResNetBlock(1, norm_inp_rpa, 64, 3, change_sample=True)
    conv_rpa = ResNetBlock(1, conv_rpa, 64, 3)
    
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3, change_sample=True)
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3)
    
    inp_rri = layers.Input(shape=(None, 1))
    norm_inp_rri = layers.Normalization()(inp_rri)
    conv_rri = ResNetBlock(1, norm_inp_rri, 64, 3, change_sample=True)
    conv_rri = ResNetBlock(1, conv_rri, 64, 3)
    
    conv_rri = ResNetBlock(1, conv_rri, 128, 3, change_sample=True)
    conv_rri = ResNetBlock(1, conv_rri, 128, 3)
    
    r_peak_features = layers.Concatenate(axis=-2)([conv_rpa, conv_rri])
    r_peak_features = ResNetBlock(1, r_peak_features, 256, 3, change_sample=True)
    r_peak_features = ResNetBlock(1, r_peak_features, 256, 3)
    r_peak_features = SEBlock()(r_peak_features)
    
    inp = layers.Input(shape=(3100, 1))  # 30s
    norm_inp = layers.Normalization()(inp)
    
    # down_sample
    ds_conv = layers.Conv1D(filters=64, kernel_size=11, strides=2, padding="same")(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=2)(ds_conv)
    
    # deep
    conv = ResNetBlock(1, ds_conv, 64, 9)
    conv = ResNetBlock(1, conv, 64, 7)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3) 
    
    r_peak_att = layers.Attention(use_scale=True)([conv, r_peak_features, r_peak_features])
    
    # bottle-neck lstm
    conv_bn1 = layers.Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(r_peak_att)
    conv_bn1 = layers.BatchNormalization()(conv_bn1)
    conv_bn1 = layers.Activation("relu")(conv_bn1)
    
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(conv_bn1)
    
    # restore  
    conv_r1 = layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding="same")(rnn)
    conv_r1 = layers.BatchNormalization()(conv_r1)
    conv_r1 = layers.Activation("relu")(conv_r1)
    conv_r1 = layers.Add()([conv, conv_r1])  # residual connection
    conv_r1 = layers.Activation("relu")(conv_r1)
    
    conv_r1 = ResNetBlock(1, conv_r1, 512, 3, change_sample=True)
    conv_r1 = ResNetBlock(1, conv_r1, 512, 3)
    
    # bottle-neck attention
    conv_bn2 = layers.Conv1D(filters=128, kernel_size=1, padding="same")(conv_r1)
    conv_bn2 = layers.BatchNormalization()(conv_bn2)
    conv_bn2 = layers.Activation("relu")(conv_bn2)
    
    att = MyAtt(depth=64, num_heads=16, seq_len=97)(conv_bn2, conv_bn2, conv_bn2)
    full = layers.Conv1D(filters=512, kernel_size=1, padding="same")(att)
    full = layers.BatchNormalization()(full)
    full = layers.Activation("relu")(full)
    full = layers.Add()([full, conv_r1])  # residual connection
    full = layers.Activation("relu")(full)
    
    conv_r2 = ResNetBlock(1, full, 1024, 3, change_sample=True)
    conv_r2 = ResNetBlock(1, conv_r2, 1024, 3)
    
    se2 = SEBlock()(conv_r2)
    
    # full segment
    out_f = layers.GlobalAvgPool1D()(se2)
    out_f = layers.Dense(1024)(out_f)
    out_f = layers.BatchNormalization()(out_f)
    out_f = layers.Activation("relu")(out_f)
    out_f = layers.Dense(1, activation="sigmoid", name="full")(out_f)

    
    model = Model(inputs=[inp, inp_rpa, inp_rri], outputs=out_f)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = {
            "full": "binary_crossentropy",
        },
        metrics = {
            "full": [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
        }
    )

    return model

model = create_model()
show_params(model, "ecg_stage")

weights_path = path.join("weights", "stage.weights.h5")
# model.save_weights(weights_path)

epochs = 100 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 128
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 50,
    patience = 5,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
    monitor = "val_full_loss",
    mode = "min",
)

cb_lr = cbk.ReduceLROnPlateau(monitor='val_full_loss', mode="min", factor=0.2, patience=10, min_lr=0.00001)

for i_fold in range(folds):
    seg_len = 30
    
    ecgs = []
    spo2s = []
    labels = []
    rpa = []
    rri = []
    p_list = np.load(path.join("gen_data", f"fold_{i_fold}_train.npy"))
    
    last_p = 0

    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, 1::])
        # raw_label = raw_label[10:-10:]

        sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=1000)
        label = divide_signal(raw_label, win_size=seg_len, step_size=10)

        if p == p_list[-2]:
            last_p = sum([len(x) for x in ecgs])

        ecgs.append(sig)
        labels.append(label)

    scaler = MinMaxScaler()

    ecgs = np.vstack(ecgs)

    ecgs = scaler.fit_transform(ecgs.T).T
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    rpa, rri = calc_ecg(ecgs, splr=100, duration=seg_len+1)
    labels = np.vstack(labels)
    mean_labels = np.mean(labels, axis=-1)
    full_labels = np.round(mean_labels)
    
    print(f"Total samples: {len(labels)}")    
    print(f"\nFold {i_fold}\n")

    total_samples = len(ecgs)
    indices = np.arange(total_samples)
    indices = np.random.permutation(indices)
    train_size = last_p 
    val_size = total_samples - last_p

    train_indices = indices[:train_size:]
    val_indices = indices[train_size:train_size+val_size:]

    print(f"Train - Val: {train_size} - {val_size}")
    class_counts = np.unique(full_labels[train_indices], return_counts=True)[1]
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")

    sample_weights = np.ones(shape=mean_labels.shape)
    sample_weights += mean_labels

    if "train" in sys.argv:
        model.fit(
            [ecgs[train_indices], rpa[train_indices], rri[train_indices]],
            full_labels[train_indices],
            epochs = epochs,
            validation_data = (
                [ecgs[val_indices], rpa[val_indices], rri[val_indices]],
                full_labels[val_indices]
            ),
            batch_size = batch_size,
            callbacks = [cb_early_stopping, cb_checkpoint, cb_lr],
            sample_weight = sample_weights[train_indices],
        )
        # model.load_weights(weights_path)

    ecgs = []
    spo2s = []
    labels = []
    rpa = []
    rri = []
    p_list = np.load(path.join("gen_data", f"fold_{i_fold}_test.npy"))
    print("\nTest result\n")
    
    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, 1::])
        offset = count_first_ele(raw_label)
        raw_sig = raw_sig[offset::]
        raw_label = raw_sig[offset::]
        # raw_label = raw_label[10:-10:]

        sig = divide_signal(raw_sig, win_size=seg_len*100)
        label = divide_signal(raw_label, win_size=seg_len)

        scaler = MinMaxScaler()

        ecgs = np.array(sig)
        ecgs = scaler.fit_transform(ecgs.T).T
        labels = np.array(label)
        ecgs = np.array([clean_ecg(e) for e in ecgs])
        rpa, rri = calc_ecg(ecgs, splr=100, duration=seg_len)
        
        labels = np.array(label)
        mean_labels = np.mean(labels, axis=-1)
        full_labels = np.round(mean_labels)

        class_counts = np.unique(full_labels, return_counts=True)[1]

        print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}")
        raw_preds = model.predict([ecgs, rpa, rri], batch_size=batch_size)
        full_preds = raw_preds
        full_preds = full_preds.flatten()

        np.save(path.join("history", f"ecg_stage_res_p{p}"), np.stack([full_labels, full_preds], axis=0))
        print(f"\nBenh nhan {p}\n")
        show_res(full_labels, np.round(full_preds))
        print()
