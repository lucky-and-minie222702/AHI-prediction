from numpy import indices
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
    inp = layers.Input(shape=(188, 8))
    
    conv = ResNetBlock(1, inp, 64, 3, kernel_regularizer=reg.l2(0.001))
    conv = ResNetBlock(1, conv, 64, 3, kernel_regularizer=reg.l2(0.001))
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True, kernel_regularizer=reg.l2(0.001))
    conv = ResNetBlock(1, conv, 128, 3, kernel_regularizer=reg.l2(0.001))
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True, kernel_regularizer=reg.l2(0.001))
    conv = ResNetBlock(1, conv, 256, 3, kernel_regularizer=reg.l2(0.001))
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True, kernel_regularizer=reg.l2(0.001))
    conv = ResNetBlock(1, conv, 512, 3, kernel_regularizer=reg.l2(0.001))
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    fc = SEBlock()(conv)
    fc = layers.GlobalAvgPool1D()(fc)
    fc = layers.Dense(512)(fc)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(rate=0.1)(fc)
    fc = layers.Activation("relu")(fc)
    out = layers.Dense(1, activation="sigmoid")(fc)
    
    model = Model(
        inputs = inp,
        outputs = out,
    )
    
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = [metrics.BinaryAccuracy(name=f"t=0.{t}", threshold=t/10) for t in range(1, 10)]
    )
    
    return model

model = create_model() 
show_params(model, "ecg_encoder + projection_head")
weights_path = path.join("res", "ecg_encoder.weights.h5")
model.save_weights(weights_path)

epochs = 100 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 512
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 50,
    patience = 10,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
)
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_encoder"))
cb_lr = WarmupCosineDecayScheduler(target_lr=0.001, warmup_epochs=10, total_epochs=epochs, min_lr=1e-6)
# cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=20, min_lr=1e-5)

ecgs = np.load(path.join("gen_data", "merged_ecgs.npy"))
labels = np.load(path.join("gen_data", "merged_labels.npy"))
indices = np.arange(len(labels))
indices = downsample_indices_manual(labels)
np.random.shuffle(indices)
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(22022009))
train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=np.random.randint(22022009))

total_samples = len(labels)
print(f"Total samples: {total_samples}\n")

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")

start_time = timer()
model.fit(
    ecgs[train_indices],
    labels[train_indices],
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (ecgs[val_indices], labels[val_indices]),
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_checkpoint],
)
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")

pred = model.predict(ecgs[test_indices], batch_size=batch_size)

res_file = open(path.join("history", "ecg_ah_res.txt"), "w")
sys.stdout = Tee(res_file)

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")

for t in np.linspace(0, 1, 11)[1:-1:]:
    t = round(t, 3)
    print(f"Threshold {t}:")
    r_pred = round_bin(pred, threshold=t)
    print_classification_metrics(labels[test_indices], r_pred)
    	
Tee.reset()