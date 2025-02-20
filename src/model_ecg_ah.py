from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

show_gpus()

folds = 1
        
def create_model():
    #######
    # ECG #
    #######
    
    inp = layers.Input(shape=(3100, 1))  # 30s
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
    
    # # bottle-neck lstm
    # conv_bn1 = layers.Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(r_peak_merge)
    # conv_bn1 = layers.BatchNormalization()(conv_bn1)
    # conv_bn1 = layers.Activation("relu")(conv_bn1)
    
    # rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(conv_bn1)
    # rnn = layers.SpatialDropout1D(rate=0.1)(rnn)
    # rnn = layers.Normalization()(rnn)
    
    # # restore  
    # conv_r1 = layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding="same")(rnn)
    # conv_r1 = layers.BatchNormalization()(conv_r1)
    # conv_r1 = layers.Activation("relu")(conv_r1)
    # conv_r1 = layers.Add()([conv, conv_r1])  # residual connection
    # conv_r1 = layers.Activation("relu")(conv_r1)
    
    # conv_r1 = ResNetBlock(1, conv_r1, 512, 3, change_sample=True)
    # conv_r1 = ResNetBlock(1, conv_r1, 512, 3)
    
    # conv_r1 = layers.SpatialDropout1D(rate=0.1)(conv_r1)
    
    # # bottle-neck attention
    # conv_bn2 = layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv_r1)
    # conv_bn2 = layers.BatchNormalization()(conv_bn2)
    # conv_bn2 = layers.Activation("relu")(conv_bn2)
    
    # att = MyAtt(depth=64, num_heads=16, dropout_rate=0.1)(conv_bn2, conv_bn2, conv_bn2)
    # full = layers.Conv1D(filters=512, kernel_size=1, padding="same")(att)
    # full = layers.BatchNormalization()(full)
    # full = layers.Activation("relu")(full)
    # full = layers.Add()([full, conv_r1])  # residual connection
    # full = layers.Activation("relu")(full)
    
    # conv_r2 = ResNetBlock(1, full, 1024, 3, change_sample=True)
    # conv_r2 = ResNetBlock(1, conv_r2, 1024, 3)
    
    # conv_r2 = layers.SpatialDropout1D(rate=0.1)(conv_r2)
    
    # se1 = SEBlock()(conv_r2)
    # se2 = SEBlock()(conv_r2)
    
    se1 = SEBlock()(conv)
    # se2 = SEBlock()(conv)
    
    # single second
    out_s = layers.GlobalAvgPool1D()(se1)
    final_out_s = layers.Dense(1, activation="sigmoid", name="single")(out_s)
    
    # full segment
    # out_f = layers.GlobalAvgPool1D()(se2)
    # final_out_f = layers.Dense(1, activation="sigmoid", name="full")(out_f)

    
    model = Model(inputs=inp, outputs=final_out_s)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001), 
        loss = { 
            # "full": "binary_crossentropy",
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

epochs = 400 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 256
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 300,
    patience = 10,
    monitor = "val_single_loss",
    mode = "min",
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
    monitor = "val_single_loss",
    mode = "min",
)

cb_lr = WarmupCosineDecayScheduler(warmup_epochs=10, total_epochs=400, target_lr=0.001, min_lr=1e-6)

for i_fold in range(folds):
    seg_len = 30
    
    ecgs = []
    labels = []
    rpa = []
    rri = []
    p_list = np.load(path.join("gen_data", f"fold_{i_fold}_train.npy"))

    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
        sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=1000)
        label = divide_signal(raw_label, win_size=(seg_len+1), step_size=10)

        ecgs.append(sig)
        labels.append(label)

    scaler = MinMaxScaler()

    ecgs = np.vstack(ecgs)
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    ecgs = scaler.fit_transform(ecgs.T).T
    
    labels = np.vstack(labels)
    labels = np.vstack([labels, labels, labels, labels, labels, labels])
    mean_labels = np.mean(labels, axis=-1)
    full_labels = np.round(mean_labels)
    single_labels = np.array([l[15] for l in labels])
    

    print(f"Total samples: {len(labels)}")
    print(f"\nFold {i_fold}\n")
    
    total_samples = len(ecgs)
    indices = np.arange(total_samples)
    indices = np.random.permutation(indices)
    np.random.shuffle(indices)
    train_size = int(total_samples * 0.8)
    val_size = total_samples - train_size

    train_indices = indices[:train_size:]
    val_indices = indices[train_size::]

    print(f"Train - Val: {train_size} - {val_size}")
    class_counts = np.unique(single_labels[train_indices], return_counts=True)[1]
    print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")

    sample_weights = [0.5 if x == 0 else 1 for x in single_labels]
    sample_weights += mean_labels
    sample_weights = np.array(sample_weights)
    
    
    if "train" in sys.argv:
        train_generator = MIOECGGenerator(X_list=[ecgs[train_indices]], y_list=[single_labels[train_indices]], batch_size=batch_size, augment_fn=my_ecg_augmentation, sample_weights=[sample_weights[train_indices]]).as_dataset()
        val_generator = MIOECGGenerator(X_list=[ecgs[val_indices]], y_list=[single_labels[val_indices]], batch_size=batch_size, augment_fn=my_ecg_augmentation, sample_weights=[sample_weights[val_indices]]).as_dataset()
        
        steps_per_epoch = len(ecgs) // batch_size
        validation_steps = len(single_labels) // batch_size
        
        model.fit(
            train_generator,
            epochs = epochs,
            validation_data = val_generator,
            batch_size = batch_size,
            callbacks = [cb_early_stopping, cb_lr, cb_checkpoint],
            
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
        
        res_file = open(path.join("history", "ah_res.txt"), "w")
        res_file.close()
        
        # model.load_weights(weights_path)
    
    ecgs = []
    labels = []
    rpa = []
    rri = []
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

        labels = np.array(label)
        mean_labels = np.mean(labels, axis=-1)
        full_labels = np.round(mean_labels)
        single_labels = np.array([l[15] for l in labels])

        class_counts = np.unique(single_labels, return_counts=True)[1]
        print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}")
        raw_preds = model.predict(ecgs, batch_size=batch_size)
        single_preds = raw_preds
        single_preds = single_preds.flatten()

        np.save(path.join("history", f"ecg_ah_res_p{p}"), np.stack([single_labels, single_preds], axis=0))
        
        res_file = open(path.join("history", "ah_res.txt"), "a")
        print(f"\nBenh nhan {p}\n")
        print(f"\nBenh nhan {p}\n", file=res_file)
        for t in np.linspace(0, 1, 11)[1:-1:]:
            t = round(t, 3)
            print(f"Threshold {t}: {acc_bin(single_labels, round_bin(single_preds, t))}")
            print(f"Threshold {t}: {acc_bin(single_labels, round_bin(single_preds, t))}", file=res_file)
        print()
        
        res_file.close()