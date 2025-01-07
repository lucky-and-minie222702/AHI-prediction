from data_functions import *
from model_functions import *

# rpa = 10 (max 120 beats, rri = 9)
def create_model():
    inp = layers.Input(shape=(3000, 1))
    en = layers.Normalization()(inp)
    
    en = ResNetBlock(1, en, 512, 9, True)
    en = ResNetBlock(1, en, 512, 9)
    en = ResNetBlock(1, en, 256, 7, True)
    en = ResNetBlock(1, en, 256, 7)
    en = ResNetBlock(1, en, 128, 5, True)
    en = ResNetBlock(1, en, 128, 5,)
    en = ResNetBlock(1, en, 64, 3, True)
    en = ResNetBlock(1, en, 64, 3,)
    en = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))(en)
    en = layers.Dense(128, activation="sigmoid")(en)  # 47 -> 32
    
    expanded_en = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(en)
    de = ResNetBlock(1, expanded_en, 64, 3, True, True)
    de = ResNetBlock(1, expanded_en, 64, 3, False, True)
    de = ResNetBlock(1, de, 128, 5, True, True)
    de = ResNetBlock(1, de, 128, 5, False, True)
    de = ResNetBlock(1, de, 256, 7, True, True)
    de = ResNetBlock(1, de, 256, 7, False, True)
    de = ResNetBlock(1, de, 512, 9, True, True)
    de = ResNetBlock(1, de, 512, 9, False, True)
    de = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))(de)
    de = layers.Dense(3000, activation="sigmoid", name="ecg")(de)
    
    de_rpa = ResNetBlock(1, expanded_en, 64, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 64, 3)
    de_rpa = ResNetBlock(1, de_rpa, 128, 5, True)
    de_rpa = ResNetBlock(1, de_rpa, 128, 5)
    de_rpa = ResNetBlock(1, de_rpa, 256, 7, True)
    de_rpa = ResNetBlock(1, de_rpa, 256, 7)
    de_rpa = layers.Flatten()(de_rpa)
    de_rpa = layers.Dense(256, activation="relu")(de_rpa)
    de_rpa = layers.Dense(60, name="rpa")(de_rpa)
    
    de_rri = ResNetBlock(1, expanded_en, 64, 3, True)
    de_rri = ResNetBlock(1, de_rri, 64, 3)
    de_rri = ResNetBlock(1, de_rri, 128, 5, True)
    de_rri = ResNetBlock(1, de_rri, 128, 5)
    de_rri = ResNetBlock(1, de_rri, 256, 7, True)
    de_rri = ResNetBlock(1, de_rri, 256, 7)
    de_rri = layers.Flatten()(de_rri)
    de_rri = layers.Dense(256, activation="relu")(de_rri)
    de_rri = layers.Dense(60, name="rri")(de_rri)
    
    decoder = Model(
        inputs = inp,
        outputs = [de, de_rpa, de_rri],
    )
    
    encoder = Model(
        inputs = inp,
        outputs = en,
    )
    
    return decoder, encoder

save_path = path.join("res", "model_auto_encoder_ECG.weights.h5")
max_epochs = 1 if "test_save" in sys.argv else 500
batch_size = 64
if "batch_size" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("batch_size")+1])

# callbacks
early_stopping_epoch = 100
if "ese" in sys.argv:
    early_stopping_epoch = int(sys.argv[sys.argv.index("ese")+1])
cb_early_stopping = cbk.EarlyStopping(
    monitor = "ecg_loss",
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 5,
)
cb_timer = TimingCallback()

if "train" in sys.argv:
    decoder, encoder = create_model()
    
    decoder.compile(
        optimizer = "adam",
        loss = {
            "ecg": "mse",
            "rpa": "mse",
            "rri": "mse",
        }
    )
    
    decoder.summary()
    sequences = np.load(path.join("patients", "merged_ECG.npy"))
    print(f"Train size: {len(sequences)}")
    rpa, rri = calc_ecg(sequences)
    hist = decoder.fit(
        sequences,
        [sequences, rpa, rri],
        epochs = max_epochs,
        batch_size = batch_size,
        callbacks = [
            cb_timer,
        ]
    )
    
    encoder.save_weights(save_path)

    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{key}_ECG_autoencoder")
        np.save(his_path, data)

    print("Saving history done!")
