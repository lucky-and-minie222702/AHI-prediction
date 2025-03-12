from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

show_gpus()

def create_model():
    inp = layers.Input(shape=(3000, 1))
    
    conv = ResNetBlock(1, inp, 64, 10, kernel_regularizer=reg.l2(0.001), activation=layers.LeakyReLU(0.3))
    conv = ResNetBlock(1, conv, 64, 10, kernel_regularizer=reg.l2(0.001), activation=layers.LeakyReLU(0.3))
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = layers.Conv1D(filters=64, kernel_size=10, strides=10, kernel_regularizer=reg.l2(0.001))(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU(0.3)(conv)
    
    conv = ResNetBlock(1, inp, 128, 10, kernel_regularizer=reg.l2(0.001), activation=layers.LeakyReLU(0.3))
    conv = ResNetBlock(1, conv, 128, 10, kernel_regularizer=reg.l2(0.001), activation=layers.LeakyReLU(0.3))
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = layers.Conv1D(filters=128, kernel_size=10, strides=10, kernel_regularizer=reg.l2(0.001))(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU(0.3)(conv)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    # attention
    att = MyAtt(depth=64, num_heads=8, dropout_rate=0.1)(conv, conv, conv)
    
    # fc
    fc = SEBlock()(att  )
    fc = layers.GlobalAvgPool1D()(fc)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg.l2(0.001))(fc)
    
    model = Model(
        inputs = inp,
        outputs = out,
    )
    
    model.compile(
        optimizer = optimizers.Adam(0.001),
        loss = "binary_crossentropy",
        metrics = [metrics.BinaryAccuracy(name=f"t=0.{t}", threshold=t/10) for t in range(1, 10)] + ["binary_crossentropy"]
    )
    
    return model

model = create_model() 
show_params(model, "ecg_ah")
weights_path = path.join("res", "ecg_ah.weights.h5")
if "pre_save" in sys.argv:
    model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 512
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 50,
    patience = 10,
    monitor = "val_binary_crossentropy",
    mode = "min",
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
    monitor = "val_binary_crossentropy",
    mode = "min",
)
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_ah"))
# cb_lr = WarmupCosineDecayScheduler(target_lr=0.001, warmup_epochs=5, total_epochs=epochs, min_lr=1e-6)
cb_lr = cbk.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, monitor = "val_binary_crossentropy", mode = "min")

seg_len = 30
step_size = 5

ecgs = []
labels = []

p_list = good_p_list()

scaler = StandardScaler()

last_p = 0

for idx, p in enumerate(p_list, start=1):
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:].flatten()
    
    sig = clean_ecg(raw_sig)    
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=step_size*100)
    label = divide_signal(raw_label, win_size=seg_len, step_size=step_size)

    ecgs.append(sig)
    labels.append(label) 
 
ecgs = np.vstack(ecgs)
ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in ecgs])
labels = np.vstack(labels)
sample_weights = np.mean(labels, axis=-1)
sample_weights += 1
labels = np.array([
    1 if np.count_nonzero(l == 1) >= 10 else 0 for l in labels
])

augment_funcs = [lambda x: add_noise(x, noise_std=0.8)]
ecgs, labels = augment_data(ecgs, augment_funcs, labels)

indices = np.arange(len(labels))
indices = downsample_indices_manual(labels)
np.random.shuffle(indices)
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(22022009))

np.save(path.join("history", "train_indices"), train_indices)
np.save(path.join("history", "test_indices"), test_indices)

train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=np.random.randint(22022009))

total_samples = len(labels)
print(f"Total samples: {total_samples}\n")

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")
print("Train - Test - Val")
print_class_counts(labels[train_indices])
print_class_counts(labels[test_indices])
print_class_counts(labels[val_indices])

train_ecgs, train_labels = ecgs[train_indices], labels[train_indices]
test_ecgs, test_labels = ecgs[test_indices], labels[test_indices]
val_ecgs, val_labels = ecgs[val_indices], labels[val_indices]

# train_ecgs, train_labels = augment_data(
#     train_ecgs, 
#     augment_funcs,
#     train_labels
# )

# test_ecgs, test_labels = augment_data(
#     test_ecgs, 
#     augment_funcs,
#     test_labels
# )

# val_ecgs, val_labels = augment_data(
#     val_ecgs, 
#     augment_funcs,
#     val_labels
# )

start_time = timer()
hist = model.fit(
    train_ecgs,
    train_labels,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (val_ecgs, val_labels),
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_checkpoint],
)
hist = hist.history
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")

pred = model.predict(test_ecgs, batch_size=batch_size)
np.save(path.join("history", "ecg_ah_predontest"), np.stack([pred.flatten(), test_labels.flatten()], axis=1))

res_file = open(path.join("history", "ecg_ah_res.txt"), "w")
sys.stdout = Tee(res_file)

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")

for t in np.linspace(0, 1, 11)[1:-1:]:
    t = round(t, 3)
    print(f"Threshold {t}:")
    r_pred = round_bin(pred, threshold=t)
    print_classification_metrics(test_labels, r_pred)
    	
Tee.reset()