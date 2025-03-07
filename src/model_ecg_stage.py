from numpy import indices
from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

show_gpus()

def create_model():
    inp = layers.Input(shape=(None, 1))
    
    encoder = get_encoder(kernel_regularizer=reg.l2(0.001))
    
    encoded_inp = encoder(inp)

    ds_conv = layers.Conv1D(filters=64, kernel_size=7, strides=2, kernel_regularizer=reg.l2(0.001))(encoded_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    
    conv = ResNetBlock(1, ds_conv, 64, 3, kernel_regularizer=reg.l2(0.001))
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
    
    fc = SEBlock(kernel_regularizer=reg.l2(0.001))(conv)
    fc = layers.GlobalAvgPool1D()(fc)
    fc = layers.Dense(512, kernel_regularizer=reg.l2(0.001))(fc)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(rate=0.1)(fc)
    fc = layers.Activation("relu")(fc)
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
show_params(model, "ecg_stage")
weights_path = path.join("res", "ecg_stage.weights.h5")
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
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_stage"))
# cb_lr = WarmupCosineDecayScheduler(target_lr=0.001, warmup_epochs=5, total_epochs=epochs, min_lr=1e-6)
cb_lr = cbk.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, monitor = "val_binary_crossentropy", mode = "min")

ecgs = np.load(path.join("gen_data", "merged_ecgs.npy"))
sample_weights = np.load(path.join("gen_data", "merged_wakes.npy")) 
labels = np.round(sample_weights)
sample_weights += 1
# ecgs, labels = dummy_data(40000)

indices = np.arange(len(labels))
indices = downsample_indices_manual(labels)
np.random.shuffle(indices)
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(22022009))
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
    ecgs[train_indices],
    labels[train_indices],
    epochs = epochs,
    batch_size = batch_size,
    sample_weight = sample_weights[train_indices],
    validation_data = (ecgs[val_indices], labels[val_indices]),
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_checkpoint],
)
hist = hist.history
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")

pred = model.predict(ecgs[test_indices], batch_size=batch_size)
np.save(path.join("history", "ecg_stage_predontest"), np.stack([pred.flatten(), labels[test_indices].flatten()], axis=1))

res_file = open(path.join("history", "ecg_stage_res.txt"), "w")
sys.stdout = Tee(res_file)

print(f"Train - Val: {len(train_indices)} - {len(val_indices)}")
print(f"Test size: {len(test_indices)}")

for t in np.linspace(0, 1, 11)[1:-1:]:
    t = round(t, 3)
    print(f"Threshold {t}:")
    r_pred = round_bin(pred, threshold=t)
    print_classification_metrics(labels[test_indices], r_pred)
    	
Tee.reset()

plt.plot(hist["binary_crossentropy"], label="loss")
plt.plot(hist["val_binary_crossentropy"], label="val_loss")
plt.legend()
plt.grid()
plt.savefig(path.join("history", "ecg_stage_plot_loss.png"))
plt.close()

plt.plot(hist["t=0.5"], label="accuracy")
plt.plot(hist["val_t=0.5"], label="val accuracy")
plt.legend()
plt.grid()
plt.savefig(path.join("history", "ecg_stage_plot_acc.png"))
plt.close()