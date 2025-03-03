from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler

show_gpus()

def augment_ecg(signal):
    # signal = time_warp(signal, sigma=0.075)
    signal = add_noise(signal, noise_std=0.1)
    # signal = time_shift(signal, shift_max=20)
    # signal *= np.random.randint(80, 120) / 100
    return signal

        
def create_model():
    inp = layers.Input(shape=(600, 10))
    en = layers.Normalization()(inp)
    
    en = layers.Conv1D(filters=64, kernel_size=11, strides=2)(en)
    en = layers.BatchNormalization()(en)
    en = layers.Activation("relu")(en)
    en = layers.MaxPool1D(pool_size=3, strides=2)(en)

    en = ResNetBlock(1, en, 64, 3, True)
    en = ResNetBlock(1, en, 64, 3)
    en = ResNetBlock(1, en, 64, 3)
       
    en = ResNetBlock(1, en, 128, 3, True)
    en = ResNetBlock(1, en, 128, 3)
    en = ResNetBlock(1, en, 128, 3)
    en = ResNetBlock(1, en, 128, 3)
    
    en = ResNetBlock(1, en, 256, 3, True)
    en = ResNetBlock(1, en, 256, 3)
    en = ResNetBlock(1, en, 256, 3)
    en = ResNetBlock(1, en, 256, 3)
    
    en = ResNetBlock(1, en, 512, 3, True)
    en = ResNetBlock(1, en, 512, 3)
    en = ResNetBlock(1, en, 512, 3)
    en = ResNetBlock(1, en, 512, 3)
    
    en = ResNetBlock(1, en, 1024, 3, True)
    en = ResNetBlock(1, en, 1024, 3)
    en = ResNetBlock(1, en, 1024, 3)

    en = SEBlock()(en)
    en = layers.Flatten()(en)
    expanded_en = layers.Reshape((640, 8))(en)

    de = ResNetBlock(1, expanded_en, 64, 3, True)
    de = ResNetBlock(1, de, 64, 3)
    de = ResNetBlock(1, de, 64, 3)
    
    de = ResNetBlock(1, de, 128, 3, True)
    de = ResNetBlock(1, de, 128, 3)
    de = ResNetBlock(1, de, 128, 3)
    
    de = ResNetBlock(1, de, 256, 3, True)
    de = ResNetBlock(1, de, 256, 3)
    de = ResNetBlock(1, de, 256, 3)
    
    de = ResNetBlock(1, de, 512, 3, True)
    de = ResNetBlock(1, de, 512, 3)
    de = ResNetBlock(1, de, 512, 3)
    
    de = ResNetBlock(1, de, 1024, 3, True)
    de = ResNetBlock(1, de, 1024, 3)
    de = ResNetBlock(1, de, 1024, 3)
    
    de = layers.Dense(512)(de)
    de = layers.BatchNormalization()(de)
    de = layers.Activation("relu")(de)
    de = layers.Dense(300)(de)
    de = layers.Flatten(name="ecg")(de)
    
    de_rpa = ResNetBlock(1, expanded_en, 64, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 64, 3)
    de_rpa = ResNetBlock(1, de_rpa, 64, 3)

    de_rpa = ResNetBlock(1, de_rpa, 128, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 128, 3)
    de_rpa = ResNetBlock(1, de_rpa, 128, 3)

    de_rpa = ResNetBlock(1, de_rpa, 256, 3, True)
    de_rpa = ResNetBlock(1, de_rpa, 256, 3)
    de_rpa = ResNetBlock(1, de_rpa, 256, 3)
    
    de_rpa = SEBlock()(de_rpa)
    de_rpa = layers.GlobalAvgPool1D()(de_rpa)
    de_rpa = layers.Dense(180, name="rpa")(de_rpa)
    
    de_rri = ResNetBlock(1, expanded_en, 64, 3, True)
    de_rri = ResNetBlock(1, de_rri, 64, 3)
    de_rri = ResNetBlock(1, de_rri, 64, 3)

    de_rri = ResNetBlock(1, de_rri, 128, 3, True)
    de_rri = ResNetBlock(1, de_rri, 128, 3)
    de_rri = ResNetBlock(1, de_rri, 128, 3)

    de_rri = ResNetBlock(1, de_rri, 256, 3, True)
    de_rri = ResNetBlock(1, de_rri, 256, 3)
    de_rri = ResNetBlock(1, de_rri, 256, 3)

    de_rri = SEBlock()(de_rri)
    de_rri = layers.GlobalAvgPool1D()(de_rri)
    de_rri = layers.Dense(180, name="rri")(de_rri)
    
    autoencoder = Model(
        inputs = inp,
        outputs = [de, de_rpa, de_rri],
    )
    
    encoder = Model(
        inputs = inp,
        outputs = expanded_en,
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

batch_size = 256 + 128
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

seg_len = 60
step_size = 30

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
rpa, rri = calc_ecg(ecgs, splr=100, duration=60, max_rpa=180, max_rri=180)
ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in ecgs])

total_samples = len(labels)
print(f"Total samples: {total_samples}\n")

start_time = timer()
model.fit(
    np.vstack([ecgs, [augment_ecg(e) for e in ecgs]]).reshape(-1, 600, 10),
    [np.vstack([ecgs, ecgs]), np.vstack([rpa, rpa]), np.vstack([rri, rri])],
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_save_encoder],
)
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")