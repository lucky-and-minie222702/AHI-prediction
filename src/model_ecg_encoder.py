from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler

show_gpus()

def augment_ecg(signal):
    signal = time_warp(signal, sigma=0.075)
    signal = add_noise(signal, noise_std=0.05)
    signal = time_shift(signal, shift_max=20)
    signal *= np.random.randint(80, 120) / 100
    return signal

        
def create_model():
    inp = layers.Input(shape=(3000, 1))
    en = layers.Normalization()(inp)
    
    en = layers.Conv1D(filters=64, kernel_size=7, strides=2)(en)
    en = layers.BatchNormalization()(en)
    en = layers.Activation("relu")(en)
    en = layers.MaxPool1D(pool_size=3, strides=2)(en)

    en = ResNetBlock(1, en, 64, 3)
    en = ResNetBlock(1, en, 64, 3)
    en = ResNetBlock(1, en, 64, 3)
       
    en = ResNetBlock(1, en, 128, 3, True)
    en = ResNetBlock(1, en, 128, 3)
    en = ResNetBlock(1, en, 128, 3)
    
    en = ResNetBlock(1, en, 256, 3, True)
    en = ResNetBlock(1, en, 256, 3)
    en = ResNetBlock(1, en, 256, 3)
    
    en = ResNetBlock(1, en, 512, 3, True)
    en = ResNetBlock(1, en, 512, 3)
    en = ResNetBlock(1, en, 512, 3)
    
    en = layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(en)
    en = layers.Conv1DTranspose(filters=32, kernel_size=3, padding="same")(en)
    en = layers.Conv1DTranspose(filters=8, kernel_size=3, padding="same")(en)
    
    # en shape (188, 8)

    de = ResNetBlock(1, en, 64, 3, True, transpose=True)
    de = ResNetBlock(1, de, 64, 3, transpose=True)
    de = ResNetBlock(1, de, 64, 3, transpose=True)
    
    de = ResNetBlock(1, de, 128, 3, True, transpose=True)
    de = ResNetBlock(1, de, 128, 3, transpose=True)
    de = ResNetBlock(1, de, 128, 3, transpose=True)
    
    de = ResNetBlock(1, de, 256, 3, True, transpose=True)
    de = ResNetBlock(1, de, 256, 3, transpose=True)
    de = ResNetBlock(1, de, 256, 3, transpose=True)
    
    de = ResNetBlock(1, de, 512, 3, True, transpose=True)
    de = ResNetBlock(1, de, 512, 3, transpose=True)
    de = ResNetBlock(1, de, 512, 3, transpose=True)
    
    de = layers.Conv1D(filters=1, kernel_size=9)(de)
    de = layers.Flatten(name="ecg")(de)
    
    de_rpa = ResNetBlock(1, en, 64, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 64, 3)

    de_rpa = ResNetBlock(1, de_rpa, 128, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 128, 3)
    de_rpa = ResNetBlock(1, de_rpa, 128, 3)

    de_rpa = ResNetBlock(1, de_rpa, 256, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 256, 3)
    
    de_rpa = SEBlock()(de_rpa)
    de_rpa = layers.GlobalAvgPool1D()(de_rpa)
    de_rpa = layers.Dense(90, name="rpa")(de_rpa)
    
    de_rri = ResNetBlock(1, en, 64, 3, True)
    de_rri = ResNetBlock(1, de_rri, 64, 3)

    de_rri = ResNetBlock(1, de_rri, 128, 3, True)
    de_rri = ResNetBlock(1, de_rri, 128, 3)
    de_rri = ResNetBlock(1, de_rri, 128, 3)

    de_rri = ResNetBlock(1, de_rri, 256, 3, True)
    de_rri = ResNetBlock(1, de_rri, 256, 3)

    de_rri = SEBlock()(de_rri)
    de_rri = layers.GlobalAvgPool1D()(de_rri)
    de_rri = layers.Dense(90, name="rri")(de_rri)
    
    autoencoder = Model(
        inputs = inp,
        outputs = [de, de_rpa, de_rri],
    )
    
    encoder = Model(
        inputs = inp,
        outputs = en,
    )
    
    autoencoder.compile(
        optimizer = "adam",
        loss = {
            "ecg": "mse",
            "rri": "mse",
            "rpa": "mse",
        },
        metrics = {
            "ecg": "mae",
            "rri": "mae",
            "rpa": "mae",
        },
    )
    
    return autoencoder, encoder

model, encoder = create_model() 
show_params(model, "ecg_encoder + projection_head")
weights_path = path.join("res", "ecg_encoder.weights.h5")
model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])
epochs = 0 if "encode" in sys.argv else epochs

batch_size = 128
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 100,
    patience = 20,
)
# cb_checkpoint = cbk.ModelCheckpoint(
#     weights_path, 
#     save_best_only = True,
#     save_weights_only = True,
# )
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_encoder"))
cb_lr = WarmupCosineDecayScheduler(target_lr=0.001, warmup_epochs=10, total_epochs=epochs, min_lr=1e-6)
# cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=20, min_lr=1e-5)
cb_save_encoder = SaveEncoderCallback(encoder, weights_path)

seg_len = 30
step_size = 15

ecgs = []
labels = []

p_list = good_p_list()

scaler = StandardScaler()

for idx, p in enumerate(p_list, start=1):
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:].flatten()
    
    sig = clean_ecg(raw_sig)    
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=step_size*100)
    label = divide_signal(raw_label, win_size=seg_len, step_size=step_size)
    
    ecgs.append(sig)
    labels.append(label) 
 
ecgs = np.vstack(ecgs)
ecgs = np.vstack([ecgs, [augment_ecg(e) for e in ecgs]])
ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in ecgs])
labels = np.vstack(labels)
labels = np.vstack([labels, labels])
labels = np.array([
    1 if np.count_nonzero(l == 1) >= 10 else 0 for l in labels
])
rpa, rri = calc_ecg(ecgs, splr=100, duration=60, max_rpa=90, max_rri=90)

total_samples = len(ecgs)
print(f"Total samples: {total_samples}\n")

start_time = timer()
model.fit(
    ecgs,
    [ecgs, rpa, rri],
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_save_encoder],
)
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")

if "encode" in sys.argv:
    encoder.load_weights(weights_path)
    encoded_ecg = encoder.predict(ecgs, batch_size=batch_size)
    np.save(path.join("gen_data", "merged_ecg.py"), encoded_ecg)
    np.save(path.join("gen_data"))