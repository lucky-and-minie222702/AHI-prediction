from data_functions import *
from model_functions import *

def create_model():
    inp = layers.Input(shape=(3000, 1))
    en = layers.Normalization()(inp)

    en = ResNetBlock(1, en, 64, 9, True)
    en = ResNetBlock(1, en, 64, 9)
    en = ResNetBlock(1, en, 64, 9) 
       
    en = ResNetBlock(1, en, 128, 7, True)
    en = ResNetBlock(1, en, 128, 7)
    en = ResNetBlock(1, en, 128, 7)
    en = ResNetBlock(1, en, 128, 7)
    en = ResNetBlock(1, en, 128, 7)
    
    en = ResNetBlock(1, en, 256, 5, True)
    en = ResNetBlock(1, en, 256, 5)
    en = ResNetBlock(1, en, 256, 5)
    en = ResNetBlock(1, en, 256, 5)
    en = ResNetBlock(1, en, 256, 5)
    
    en = ResNetBlock(1, en, 512, 3, True)
    en = ResNetBlock(1, en, 512, 3)
    en = ResNetBlock(1, en, 512, 3)
    
    en = ResNetBlock(1, en, 1024, 3, True)
    en = ResNetBlock(1, en, 1024, 3)

    en = SEBlock()(en)
    en = layers.GlobalAvgPool1D()(en)
    en = layers.Flatten()(en)
    en = layers.Dense(1500)(en)
    
    expanded_en = layers.Reshape((150, 10))(en)
    de = ResNetBlock(1, expanded_en, 512, 3, True, True)
    de = ResNetBlock(1, de, 512, 3, False, True)
    de = ResNetBlock(1, de, 512, 3, False, True)
    
    de = ResNetBlock(1, de, 256, 5, True, True)
    de = ResNetBlock(1, de, 256, 5, False, True)
    de = ResNetBlock(1, de, 256, 5, False, True)
    de = ResNetBlock(1, de, 256, 5, False, True)
    de = ResNetBlock(1, de, 256, 5, False, True)
    
    de = ResNetBlock(1, de, 128, 7, True, True)
    de = ResNetBlock(1, de, 128, 7, False, True)
    de = ResNetBlock(1, de, 128, 7, False, True)
    de = ResNetBlock(1, de, 128, 7, False, True)
    de = ResNetBlock(1, de, 128, 7, False, True)
    
    de = ResNetBlock(1, de, 64, 9, True, True)
    de = ResNetBlock(1, de, 64, 9, False, True)
    de = ResNetBlock(1, de, 64, 9, False, True)
    
    de = layers.Conv1D(filters=32, kernel_size=3)(de)
    de = layers.BatchNormalization()(de)
    de = layers.LeakyReLU(negative_slope=0.25)(de)
    de = layers.Conv1D(filters=16, kernel_size=3)(de)
    de = layers.BatchNormalization()(de)
    de = layers.LeakyReLU(negative_slope=0.25)(de)
    de = layers.Conv1D(filters=8, kernel_size=3)(de)
    de = layers.BatchNormalization()(de)
    de = layers.LeakyReLU(negative_slope=0.25)(de)
    de = SEBlock()(de)

    de = layers.Flatten()(de)
    de = layers.Dense(3000, activation="sigmoid", name="ecg")(de)
    
    de_rpa = ResNetBlock(1, expanded_en, 64, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 64, 3)
    de_rpa = ResNetBlock(1, de_rpa, 64, 3)
    de_rpa = ResNetBlock(1, de_rpa, 128, 5, True)
    de_rpa = ResNetBlock(1, de_rpa, 128, 5)
    de_rpa = ResNetBlock(1, de_rpa, 128, 5)
    de_rpa = ResNetBlock(1, de_rpa, 256, 7, True)
    de_rpa = ResNetBlock(1, de_rpa, 256, 7)
    de_rpa = ResNetBlock(1, de_rpa, 256, 7)
    de_rpa = SEBlock()(de_rpa)
    de_rpa = layers.GlobalAvgPool1D()(de_rpa)
    de_rpa = layers.Dense(60, name="rpa")(de_rpa)
    
    de_rri = ResNetBlock(1, expanded_en, 64, 3, True)
    de_rri = ResNetBlock(1, de_rri, 64, 3)
    de_rri = ResNetBlock(1, de_rri, 64, 3)
    de_rri = ResNetBlock(1, de_rri, 128, 5, True)
    de_rri = ResNetBlock(1, de_rri, 128, 5)
    de_rri = ResNetBlock(1, de_rri, 128, 5)
    de_rri = ResNetBlock(1, de_rri, 256, 7, True)
    de_rri = ResNetBlock(1, de_rri, 256, 7)
    de_rri = ResNetBlock(1, de_rri, 256, 7)
    de_rri = SEBlock()(de_rri)
    de_rri = layers.GlobalAvgPool1D()(de_rri)
    de_rri = layers.Dense(60, name="rri")(de_rri)
    
    autoencoder = Model(
        inputs = inp,
        outputs = [de, de_rpa, de_rri],
    )
    
    encoder = Model(
        inputs = inp,
        outputs = en,
    )
    
    return autoencoder, encoder

save_path = path.join("res", "model_auto_encoder_ECG.weights.h5")
max_epochs = 1 if "test_save" in sys.argv else 150
batch_size = 64
if "batch_size" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("batch_size")+1])

# callbacks
early_stopping_epoch = 70
if "ese" in sys.argv:
    early_stopping_epoch = int(sys.argv[sys.argv.index("ese")+1])
cb_early_stopping = cbk.EarlyStopping(
    monitor = "val_ecg_loss",
    mode = "min",
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 5,
)
cb_timer = TimingCallback()

autoencoder, encoder = create_model()

autoencoder.compile(
    optimizer = "adam",
    loss = {
        "ecg": "mse",
        "rpa": "mse",
        "rri": "mse",
    },
    metrics = {
        "ecg": "mae",
        "rpa": "mae",
        "rri": "mae",
    },
)

# autoencoder.summary()
show_params(autoencoder, "autoencoder")

sequences = np.load(path.join("patients", "merged_ECG.npy"))
print(sequences.shape)
rpa, rri = calc_ecg(sequences)

best = np.count_nonzero(rpa, axis=1) >= 15  # min 30 bpm
rpa = rpa[best]
rri = rri[best]
sequences = sequences[best]

if "train" in sys.argv:
    print(f"Train size: {len(sequences)}")
    # sequences = pad_sequences(sequences, maxlen=3008)
    hist = autoencoder.fit(
        sequences,
        [sequences, rpa, rri],
        epochs = max_epochs,
        batch_size = batch_size,
        validation_split = 0.2,
        callbacks = [
            cb_timer,
            cb_early_stopping,
            SaveEncoderCallback(encoder, save_path),
        ]        
    )
    
    encoder.save_weights(save_path)

    t = sum(cb_timer.logs)
    print(f"Total training time: {convert_seconds(t)}")
    print(f"Total epochs: {len(cb_timer.logs)}\n")


    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{key}_ECG_autoencoder")
        np.save(his_path, data)

    print("Saving history done!")


if "encode" in sys.argv:
    del autoencoder
    encoder.load_weights(save_path)
    encoded_ECG = encoder.predict(sequences, batch_size=batch_size).squeeze()
    np.save(path.join("patients", "merged_ECG.npy"), encoded_ECG)
    print(encoded_ECG.shape)
    print("Encoding done!")