from data_functions import *
from model_functions import *
from scipy.signal import find_peaks

def create_model():
    inp = layers.Input(shape=(60, 1))
    en = layers.Normalization()(inp)
    
    en = ResNetBlock(1, en, 64, 3)
    en = ResNetBlock(1, en, 64, 3)
    en = ResNetBlock(1, en, 64, 3)
    
    en = ResNetBlock(1, en, 128, 3)
    en = ResNetBlock(1, en, 128, 3)
    en = ResNetBlock(1, en, 128, 3)
    
    en = ResNetBlock(1, en, 256, 3)
    en = ResNetBlock(1, en, 256, 3)
    en = ResNetBlock(1, en, 256, 3)
    
    en = SEBlock()(en)
    en = layers.GlobalAvgPool1D()(en)
    en = layers.Dense(128)(en)
    en = layers.Activation("relu")(en)
    en = layers.Dense(96)(en)
    
    expanded_en = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(en)
    
    de = ResNetBlock(1, expanded_en, 64, 3)
    de = ResNetBlock(1, de, 64, 3)
    de = ResNetBlock(1, de, 64, 3)

    de = ResNetBlock(1, de, 128, 3)
    de = ResNetBlock(1, de, 128, 3)
    de = ResNetBlock(1, de, 128, 3)
    
    de = ResNetBlock(1, de, 256, 3)
    de = ResNetBlock(1, de, 256, 3)
    de = ResNetBlock(1, de, 256, 3)

    de = SEBlock()(de)
    de = layers.GlobalAvgPool1D()(de)
    de = layers.Dense(128)(de)
    de = layers.BatchNormalization()(de)
    de = layers.Activation("relu")(de)
    de = layers.Dense(60, activation="sigmoid", name="spo2")(de)
    
    de_stats = ResNetBlock(1, expanded_en, 64, 3)
    de_stats = ResNetBlock(1, de_stats, 64, 3)
    de_stats = ResNetBlock(1, de_stats, 64, 3)

    de_stats = SEBlock()(de_stats)
    de_stats = layers.GlobalAvgPool1D()(de_stats)
    de_stats = layers.Dense(32)(de_stats)
    de_stats = layers.Activation("relu")(de_stats)
    de_stats = layers.Dense(7, activation="sigmoid", name="stats")(de_stats)
    
    de_peaks = ResNetBlock(1, expanded_en, 64, 3)
    de_peaks = ResNetBlock(1, de_peaks, 64, 3)
    de_peaks = ResNetBlock(1, de_peaks, 64, 3)
    
    de_peaks = ResNetBlock(1, de_peaks, 128, 3)
    de_peaks = ResNetBlock(1, de_peaks, 128, 3)
    de_peaks = ResNetBlock(1, de_peaks, 128, 3)

    de_peaks = SEBlock()(de_peaks)
    de_peaks = layers.GlobalAvgPool1D()(de_peaks)
    de_peaks = layers.Dense(64)(de_peaks)
    de_peaks = layers.Activation("relu")(de_peaks)
    de_peaks = layers.Dense(30, activation="sigmoid", name="peaks")(de_peaks)
    
    de_drops = ResNetBlock(1, expanded_en, 64, 3)
    de_drops = ResNetBlock(1, de_drops, 64, 3)
    de_drops = ResNetBlock(1, de_drops, 64, 3)
    
    de_drops = ResNetBlock(1, de_drops, 128, 3)
    de_drops = ResNetBlock(1, de_drops, 128, 3)
    de_drops = ResNetBlock(1, de_drops, 128, 3)

    de_drops = SEBlock()(de_drops)
    de_drops = layers.GlobalAvgPool1D()(de_drops)
    de_drops = layers.Dense(64)(de_drops)
    de_drops = layers.Activation("relu")(de_drops)
    de_drops = layers.Dense(30, activation="sigmoid", name="drops")(de_drops)
    
    
    autoencoder = Model(
        inputs = inp,
        outputs = [de, de_stats, de_peaks, de_drops],
    )
    
    encoder = Model(
        inputs = inp,
        outputs = en,
    )
    
    return autoencoder, encoder

save_path = path.join("res", "model_auto_encoder_SpO2.weights.h5")
max_epochs = 1 if "test_save" in sys.argv else 300
batch_size = 64
if "batch_size" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("batch_size")+1])

# callbacks
early_stopping_epoch = 50
if "ese" in sys.argv:
    early_stopping_epoch = int(sys.argv[sys.argv.index("ese")+1])
cb_early_stopping = cbk.EarlyStopping(
    monitor = "val_spo2_loss",  
    mode = "min",
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 7,
)
cb_timer = TimingCallback()

autoencoder, encoder = create_model()

autoencoder.compile(
    optimizer = "adam",
    loss = {
        "spo2": "mse",
        "stats": "mse",
        "peaks": "mse",
        "drops": "mse",
    },
    metrics = {
        "spo2": "mae",
        "stats": "mae",
        "peaks": "mae",
        "drops": "mae",
    },
)

# autoencoder.summary()
show_params(autoencoder, "autoencoder")

sequences = np.load(path.join("patients", "merged_SpO2.npy"))
stats = np.array(calc_stats(sequences))
peaks = np.array(pad_sequences([x[find_peaks(x)[0]] for x in sequences], maxlen=30))
drops = np.array(pad_sequences([x[find_peaks(-x)[0]] for x in sequences], maxlen=30))

print(sequences.shape)

if "train" in sys.argv:
    print(f"Train size: {len(sequences)}")
    # sequences = pad_sequences(sequences, maxlen=3008)
    hist = autoencoder.fit(
        sequences,
        [sequences, stats, peaks, drops],
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
        his_path = path.join("history", f"{key}_SpO2_autoencoder")
        np.save(his_path, data)

    print("Saving history done!")


if "encode" in sys.argv:
    del autoencoder
    sequences = np.load(path.join("patients", "merged_SpO2.npy"))
    encoder.load_weights(save_path)
    encoded_SpO2 = encoder.predict(sequences, batch_size=batch_size).squeeze()
    np.save(path.join("patients", "merged_SpO2.npy"), encoded_SpO2)
    print(encoded_SpO2.shape)
    print("Encoding done!")