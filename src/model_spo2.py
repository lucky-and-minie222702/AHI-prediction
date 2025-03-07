from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib as plt

show_gpus()

def augment_spo2(signal):
    signal = time_warp(signal, sigma=0.075)
    signal = add_noise(signal, noise_std=0.05)
    signal = time_shift(signal, shift_max=20)
    signal *= np.random.randint(80, 120) / 100
    return signal

        
def create_model():
    inp = layers.Input(shape=(30, 1))
    
    conv = layers.Conv1D(filters=64, kernel_size=3)(inp)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    
    conv = layers.Conv1D(filters=128, kernel_size=3)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    
    conv = layers.Conv1D(filters=256, kernel_size=3)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    
    conv = layers.Conv1D(filters=512, kernel_size=3)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    
    fc = SEBlock()(conv)
    fc = layers.GlobalAvgPool1D()(fc)
    out = layers.Dense(1, activation="sigmoid")(fc)
    
    
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
show_params(model, "spo2_ah")
weights_path = path.join("res", "spo2_ah.weights.h5")
if "pre_save" in sys.argv:
    model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

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
cb_his = HistoryAutosaver(save_path=path.join("history", "spo2_ah"))
# cb_lr = WarmupCosineDecayScheduler(target_lr=0.001, warmup_epochs=10, total_epochs=epochs, min_lr=1e-6)
cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6)

seg_len = 30
step_size = 5 

spo2s = []
labels = []

p_list = good_p_list()

scaler = StandardScaler()

for idx, p in enumerate(p_list, start=1):
    raw_sig = np.load(path.join("data", f"benhnhan{p}spo2.npy"))
    raw_label = np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:].flatten()
    
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=step_size*100)
    label = divide_signal(raw_label, win_size=seg_len, step_size=step_size)
    
    spo2s.append(sig)
    labels.append(label) 
 
spo2s = np.vstack(spo2s)
spo2s /= 100
labels = np.vstack(labels)
labels = np.array([
    1 if np.count_nonzero(l == 1) >= 10 else 0 for l in labels
])

total_samples = len(spo2s)
print(f"Total samples: {total_samples}\n")

indices = np.arange(len(labels))
indices = downsample_indices_manual(labels)
np.random.shuffle(indices)
# train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(22022009))
train_indices = np.load(path.join("history", "train_indices.npy"))
test_indices = np.load(path.join("history", "test_indices.npy"))
train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=np.random.randint(22022009))

total_samples = len(labels)
print(f"Total samples: {total_samples}\n")

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")
print("Train - Test - Val")
print_class_counts(labels[train_indices])
print_class_counts(labels[test_indices])
print_class_counts(labels[val_indices])

start_time = timer()
hist = model.fit(
    spo2s[train_indices],
    labels[train_indices],
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (spo2s[val_indices], labels[val_indices]),
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_checkpoint],
)
hist = hist.history
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")

pred = model.predict(spo2s[test_indices], batch_size=batch_size)
np.save(path.join("history", "spo2_ah_predontest"), np.stack([pred.flatten(), labels[test_indices].flatten()], axis=1))

res_file = open(path.join("history", "spo2_ah_res.txt"), "w")
sys.stdout = Tee(res_file)

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")

for t in np.linspace(0, 1, 11)[1:-1:]:
    t = round(t, 3)
    print(f"Threshold {t}:")
    r_pred = round_bin(pred, threshold=t)
    print_classification_metrics(labels[test_indices], r_pred)
    	
Tee.reset()

# plt.plot(hist["binary_crossentropy"], label="loss")
# plt.plot(hist["val_binary_crossentropy"], label="val_loss")
# plt.legend()
# plt.grid()
# plt.savefig(path.join("history", "spo2_ah_plot_loss.png"))
# plt.close()

# plt.plot(hist["t=0.5"], label="accuracy")
# plt.plot(hist["val_t=0.5"], label="val accuracy")
# plt.legend()
# plt.grid()
# plt.savefig(path.join("history", "spo2_ah_plot_acc.png"))
# plt.close()