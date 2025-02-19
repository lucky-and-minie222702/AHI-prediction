from numpy import ndim
from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

show_gpus()

folds = 3
        
def create_model():
    inp_psd = layers.Input(shape=(None, 2))
    norm_inp_psd = layers.Normalization()(inp_psd)
    
    psd_ds = layers.Conv1D(filters=32, kernel_size=5, strides=2, padding="same")(norm_inp_psd)
    psd_ds = layers.BatchNormalization()(psd_ds)
    psd_ds = layers.Activation("relu")(psd_ds)
    psd_ds = layers.MaxPool1D(pool_size=2)(psd_ds)
    
    psd_conv = ResNetBlock(1, psd_ds, 64, 3)
    psd_conv = ResNetBlock(1, psd_conv, 64, 3)
    
    psd_conv = ResNetBlock(1, psd_conv, 128, 3, change_sample=True)
    psd_conv = ResNetBlock(1, psd_conv, 128, 3)
    
    
    inp_fft = layers.Input(shape=(None, 1))
    norm_inp_fft = layers.Normalization()(inp_fft)
    
    fft_ds = layers.Conv1D(filters=32, kernel_size=5, strides=2, padding="same")(norm_inp_fft)
    fft_ds = layers.BatchNormalization()(fft_ds)
    fft_ds = layers.Activation("relu")(fft_ds)
    fft_ds = layers.MaxPool1D(pool_size=2)(fft_ds)
    
    fft_conv = ResNetBlock(1, fft_ds, 64, 3)
    fft_conv = ResNetBlock(1, fft_conv, 64, 3)
    
    fft_conv = ResNetBlock(1, fft_conv, 128, 3, change_sample=True)
    fft_conv = ResNetBlock(1, fft_conv, 128, 3)
    
    
    # merge using attention
    merge_att = MyAtt(depth=64, num_heads=8)(fft_conv, psd_conv, psd_conv)
    
    merge_conv = ResNetBlock(1, merge_att, 512, 3, change_sample=True)
    merge_conv = ResNetBlock(1, merge_conv, 512, 3)
    merge_conv = ResNetBlock(1, merge_conv, 512, 3)
    
    merge_conv = layers.SpatialDropout1D(rate=0.1)(merge_conv)
    
    se1 = SEBlock()(merge_conv)
    se2 = SEBlock()(merge_conv)
    
    fc1 = layers.GlobalAvgPool1D()(se1)
    fc2 = layers.GlobalAvgPool1D()(se2)
    
    out1 = layers.Dense(1, activation="sigmoid", name="full")(fc1)
    out2 = layers.Dense(1, activation="sigmoid", name="single")(fc2)
    
    
    
    model = Model(inputs=[inp_psd, inp_fft], outputs=[out1, out2])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
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
show_params(model, "ecg_ah")

weights_path = path.join("weights", "ah.weights.h5")
# model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

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
    monitor = "val_single_loss",
    mode = "min",
)

cb_lr = cbk.ReduceLROnPlateau(monitor='val_single_loss', mode="min", factor=0.2, patience=10, min_lr=0.00001)

for i_fold in range(folds):
    seg_len = 30
    
    ecgs = []
    labels = []
    p_list = np.load(path.join("gen_data", f"fold_{i_fold}_train.npy"))

    last_p = 0

    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])

        sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=1000)
        label = divide_signal(raw_label, win_size=(seg_len+1), step_size=10)
        
        if p == p_list[-2]:
            last_p = sum([len(x) for x in ecgs])

        ecgs.append(sig)
        labels.append(label)

    scaler = MinMaxScaler()

    ecgs = np.vstack(ecgs)
    ecgs = scaler.fit_transform(ecgs.T).T
    
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    psds = np.array([calc_psd(e) for e in ecgs])
    ffts = np.array([calc_fft(e) for e in ecgs])
    labels = np.vstack(labels)
    mean_labels = np.mean(labels, axis=-1)
    full_labels = np.round(mean_labels)
    single_labels = np.array([l[15] for l in labels])

    print(f"Total samples: {len(labels)}")
    print(f"\nFold {i_fold}\n")
    
    total_samples = len(ecgs)
    indices = np.arange(total_samples)
    indices = np.random.permutation(indices)
    train_size = last_p 
    val_size = total_samples - last_p

    train_indices = indices[:train_size:]
    val_indices = indices[train_size::]

    print(f"Train - Val: {train_size} - {val_size}")
    class_counts = np.unique(single_labels[train_indices], return_counts=True)[1]
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")

    sample_weights = [0.5 if x == 0 else 1 for x in single_labels]
    sample_weights += mean_labels
    sample_weights = np.array(sample_weights)
    
    
    if "train" in sys.argv:
        model.fit(
            [psds[train_indices], ffts[train_indices]],
            [full_labels[train_indices], single_labels[train_indices]],
            epochs = epochs,
            validation_data = (
                [psds[val_indices], ffts[val_indices]],
                [full_labels[val_indices], single_labels[val_indices]]
            ),
            batch_size = batch_size,
            callbacks = [cb_early_stopping, cb_lr],
            sample_weight = sample_weights[train_indices],
        )
        # model.load_weights(weights_path)
    
    ecgs = []
    labels = []
    p_list = np.load(path.join("gen_data", f"fold_{i_fold}_test.npy"))
    print("\nTest result\n")

    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])

        sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=100)
        label = divide_signal(raw_label, win_size=(seg_len+1), step_size=1)

        scaler = MinMaxScaler()

        ecgs = np.array(sig)
        ecgs = scaler.fit_transform(ecgs.T).T
        labels = np.array(label)
        ecgs = np.array([clean_ecg(e) for e in ecgs])
        psds = np.array([calc_psd(e) for e in ecgs])
        ffts = np.array([calc_fft(e) for e in ecgs])

        labels = np.array(label)
        mean_labels = np.mean(labels, axis=-1)
        full_labels = np.round(mean_labels)
        single_labels = np.array([l[15] for l in labels])

        class_counts = np.unique(single_labels, return_counts=True)[1]
        print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}")
        raw_preds = model.predict([psds, ffts], batch_size=batch_size)
        single_preds = raw_preds[1]
        single_preds = single_preds.flatten()

        np.save(path.join("history", f"ecg_ah_res_p{p}"), np.stack([single_labels, single_preds], axis=0))
        print(f"\nBenh nhan {p}\n")
        show_res(single_labels, np.round(single_preds))
        print()
