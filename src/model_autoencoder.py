from idna import encode
from model_functions import *
from data_functions import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


def  create_model():
    inp = layers.Input(shape=(3100, 1))
    norm_inp = layers.Normalization()(inp)

    ds_conv = layers.Conv1D(filters=32, kernel_size=7, strides=2, padding="same")(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=2)(ds_conv)
    
    conv = ResNetBlock(1, ds_conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    
    conv = ResNetBlock(1, conv, 1024, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 1024, 3)
    conv = ResNetBlock(1, conv, 1024, 3)
    
    encode_out = layers.Conv1D(filters=256, kernel_size=3)(conv)
    encode_out = layers.BatchNormalization()(encode_out)
    encode_out = layers.Activation("relu")(encode_out)
    encode_out = layers.Conv1D(filters=64, kernel_size=3)(encode_out)
    encode_out = layers.BatchNormalization()(encode_out)
    encode_out = layers.Reshape((160, 18))(encode_out)
    encode_out = layers.Normalization()(encode_out)
    
    print(encode_out.shape)
    
    conv_r = ResNetBlock(1, encode_out, 64, 3, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 64, 3, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 64, 3, activation=layers.LeakyReLU(alpha=0.3))
    
    conv_r = ResNetBlock(1, conv_r, 128, 3, change_sample=True, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 128, 3, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 128, 3, activation=layers.LeakyReLU(alpha=0.3))
    
    conv_r = ResNetBlock(1, conv_r, 256, 3, change_sample=True, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 256, 3, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 256, 3, activation=layers.LeakyReLU(alpha=0.3))
    
    conv_r = ResNetBlock(1, conv_r, 512, 3, change_sample=True, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 512, 3, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 512, 3, activation=layers.LeakyReLU(alpha=0.3))
    
    conv_r = ResNetBlock(1, conv_r, 1024, 3, change_sample=True, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 1024, 3, activation=layers.LeakyReLU(alpha=0.3))
    conv_r = ResNetBlock(1, conv_r, 1024, 3, activation=layers.LeakyReLU(alpha=0.3))
    
    out = layers.Conv1D(filters=310, kernel_size=3, padding="same")(conv_r)
    # out = layers.Normalization()(out)
    out = layers.Activation("sigmoid")(out)
    out = layers.Flatten()(out)
    
    print(out.shape)
    

    model = Model(inputs=inp, outputs=out)
    encoder = Model(inputs=inp, outputs=encode_out)
    show_params(model)
    return model, encoder
    
model, encoder = create_model()
weights_path = path.join("history", "encoder.weights.h5")
autoencoder_path = path.join("history", "autoencoder.weights.h5")
encoder.save_weights(weights_path)
model.save_weights(autoencoder_path)


model.compile(
    loss = "mse",
    metrics = ["mae"],
    weighted_metrics = [],
)


epochs = 300
batch_size = 256


cb_his = HistoryAutosaver(path.join("history", "ecg_autoencoder"))
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 100,
    patience = 10,
    # monitor = "val_single_loss",
    # mode = "min",
)
cb_save_encoder = SaveEncoderCallback(encoder=encoder, save_path=weights_path)
cb_lr = WarmupCosineDecayScheduler(warmup_epochs=20, total_epochs=300, target_lr=0.001, min_lr=1e-6)
cb_checkpoint = cbk.ModelCheckpoint(
    autoencoder_path,
    save_weights_only = True,
    save_best_only = True,
)


scaler = MinMaxScaler()


if "train" in sys.argv:
    # train
    # model.load_weights(weights_path)
    
    ecgs = []
    p_list = raw_p_list[:20:]
    seg_len = 30
    last_p = 0

    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
        sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=1000)
        label = divide_signal(raw_label, win_size=(seg_len+1), step_size=10)
        
        if p == p_list[-2]:
            last_p = len(ecgs)

        ecgs.append(sig)


    val_ecgs = ecgs[last_p::]
    ecgs = ecgs[:last_p:]


    ecgs = np.vstack(ecgs)
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    ecgs = np.vstack([
        ecgs,
        np.array([time_warp(e, sigma=0.2) for e in ecgs]),
        np.array([time_shift(e, shift_max=20) for e in ecgs]),
        np.array([bandpass(e, 100, 5, 35, 1) for e in ecgs]),
        np.array([bandpass(e, 100, 3, 45, 1) for e in ecgs]),
        np.array([frequency_noise(e, noise_std=0.15) for e in ecgs]),
    ])
    ecgs = scaler.fit_transform(ecgs.T).T


    # val
    val_ecgs = np.vstack(val_ecgs)
    val_ecgs = np.array([clean_ecg(e) for e in val_ecgs])
    val_ecgs = np.vstack([
        val_ecgs,
        np.array([time_warp(e, sigma=0.2) for e in val_ecgs]),
        np.array([time_shift(e, shift_max=20) for e in val_ecgs]),
        np.array([bandpass(e, 100, 5, 35, 1) for e in val_ecgs]),
        np.array([bandpass(e, 100, 3, 45, 1) for e in val_ecgs]),
        np.array([frequency_noise(e, noise_std=0.15) for e in val_ecgs]),
    ])
    val_ecgs = scaler.fit_transform(val_ecgs.T).T

    train_generator = DynamicAugmentedECGDataset(ecgs[:len(ecgs) // 6:], ecgs[:len(ecgs) // 6:],  ecgs, ecgs, batch_size=batch_size, num_augmented_versions=6).as_dataset()

    steps_per_epoch = len(ecgs) // batch_size
    steps_per_epoch //= 6


    model.fit(
        train_generator,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = (val_ecgs, val_ecgs),
        steps_per_epoch = steps_per_epoch,
        callbacks = [cb_his, cb_early_stopping, cb_lr, cb_save_encoder, cb_checkpoint]
    )


if "test" in sys.argv:
    # test
    ecgs = []
    p_list = raw_p_list[22::]
    seg_len = 30
    last_p = 0

    for p in p_list:
        raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
        raw_label = np.squeeze(np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:])
        sig = divide_signal(raw_sig, win_size=(seg_len+1)*100, step_size=1000)
        label = divide_signal(raw_label, win_size=(seg_len+1), step_size=10)
        
        if p == p_list[-2]:
            last_p = len(ecgs)

        ecgs.append(sig)
        

    ecgs = np.vstack(ecgs)
    ecgs = np.array([clean_ecg(e) for e in ecgs])
    ecgs = scaler.fit_transform(ecgs.T).T
    
    model.load_weights(autoencoder_path)

    pred = model.predict(ecgs, batch_size=batch_size)
    mae = mean_absolute_error(ecgs, pred)
    mse = mean_squared_error(ecgs, pred)

    res_file = open(path.join("history", "autoencoder.txt"), "w")
    print("MAE:", mae)
    print("MSE:", mse)
    print("MAE:", mae, file=res_file)
    print("MSE:", mse, file=res_file)
    res_file.close()
    
