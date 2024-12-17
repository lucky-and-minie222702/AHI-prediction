from model_functions import *
from data_functions import *

def create_model_ECG(name: str):    
    # 1000, 1 - 10 seconds
    inp = layers.Input(shape=(None, 1))
    conv = layers.Normalization()(inp)
    
    conv = layers.Conv1D(filters=32, kernel_size=3, kernel_regularizer=reg.L2(), padding="same")(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    
    conv = ResNetBlock(1, conv, 64, True)
    conv = ResNetBlock(1, conv, 64)
    conv = ResNetBlock(1, conv, 64)
    
    conv = SEBlock(reduction_ratio=2)(conv)
    
    conv = ResNetBlock(1, conv, 128, True)
    conv = ResNetBlock(1, conv, 128)
    conv = ResNetBlock(1, conv, 128)
    
    conv = SEBlock(reduction_ratio=4)(conv)
    
    conv = ResNetBlock(1, conv, 256, True)
    conv = ResNetBlock(1, conv, 256)
    conv = ResNetBlock(1, conv, 256)
    
    conv = SEBlock(reduction_ratio=6)(conv)
    
    
    # for stage detecting
    stage_conv = ResNetBlock(1, conv, 512, True)
    stage_conv = ResNetBlock(1, stage_conv, 512)
    stage_conv = ResNetBlock(1, stage_conv, 512)
    
    stage_conv = SEBlock(reduction_ratio=8)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 1024, True)
    stage_conv = ResNetBlock(1, stage_conv, 1024)
    stage_conv = ResNetBlock(1, stage_conv, 1024)
    
    stage_conv = SEBlock(reduction_ratio=10)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 2048, True)
    stage_conv = ResNetBlock(1, stage_conv, 2048)
    stage_conv = ResNetBlock(1, stage_conv, 2048)
    
    stage_att = SEBlock(reduction_ratio=10)(stage_conv)

    stage_flat = layers.GlobalAvgPool1D()(stage_att)
    stage_flat = layers.Flatten()(stage_flat)
    stage_out = layers.Dense(1, activation="sigmoid", name="stage")(stage_flat)
    
    
    # for apnea hyponea detecting
    ah_conv = ResNetBlock(1, conv, 512, True)
    ah_conv = ResNetBlock(1, ah_conv, 512)
    ah_conv = ResNetBlock(1, ah_conv, 512)
    
    ah_conv = SEBlock(reduction_ratio=8)(ah_conv)
    
    ah_conv = ResNetBlock(1, ah_conv, 1024, True)
    ah_conv = ResNetBlock(1, ah_conv, 1024)
    ah_conv = ResNetBlock(1, ah_conv, 1024)
    
    ah_conv = SEBlock(reduction_ratio=8)(ah_conv)
    
    ah_conv = ResNetBlock(1, ah_conv, 2048, True)
    ah_conv = ResNetBlock(1, ah_conv, 2048)
    ah_conv = ResNetBlock(1, ah_conv, 2048)
    
    ah_att = SEBlock(reduction_ratio=10)(ah_conv)

    ah_flat = layers.GlobalAvgPool1D()(ah_att)
    ah_flat = layers.Flatten()(ah_flat)
    ah_out = layers.Dense(1, activation="sigmoid", name="ah")(ah_flat)

    
    model = Model(
        inputs = inp,
        outputs = [stage_out, ah_out],
    )

    show_params(model, name)
        
    return model, stage_flat, ah_flat

save_path = path.join("res", "model_ECG.weights.h5")
model = create_model_ECG("ECG")[0]
model.compile(
    optimizer = "Adam",
    loss = "binary_crossentropy",
    metrics = {
        "stage": [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
        "ah": [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
    }
)

name = sys.argv[sys.argv.index("id")+1]

max_epochs = 200
batch_size = 32

# callbacks
early_stopping_epoch = 50
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 3,
)
cb_checkpoint = cbk.ModelCheckpoint(
    save_path, 
    save_best_only = True,
    save_weights_only = True,
)
cb_timer = TimingCallback()
lr_scheduler = cbk.ReduceLROnPlateau(
    factor = 0.5,
    min_lr = 0.000001,
    patience = 5,
)

maxlen = 13880

sequences = []
annotations = []
stages = []

for i in range(1, 26):
    seq = np.load(path.join("patients", f"patients_{i}_ECG.npy"))
    ann = np.load(path.join("patients", f"patients_{i}_anns.npy"))
    stage = np.load(path.join("patients", f"patients_{i}_stages.npy"))
    
    sequences += np.split(seq, len(seq) // 1000)
    annotations.extend(ann.tolist())
    stages.extend(stage.tolist())

sequences = np.array(sequences)
annotations = np.array(annotations)

sequences = np.vstack(
    [sequences, sequences + np.random.normal(0, 0.003, sequences.shape), add_baseline_wander(sequences, frequency=0.05, amplitude=0.05, sampling_rate=100)]
)
annotations = np.concatenate(
    [annotations, annotations, annotations]
)
stages = np.concatenate(
    [stages, stages, stages]
)

indices = np.arange(len(sequences))
train_indices, test_indices = train_test_split(indices, test_size=0.2,random_state=np.random.randint(22022009))

X_train = sequences[train_indices]
y_stage_train = stages[train_indices]
y_ah_train = annotations[train_indices]
X_test = sequences[test_indices]
y_stage_test = stages[test_indices]
y_ah_test = annotations[test_indices]

print(f"Train size: {X_train.shape[0]} - Test size: {X_test.shape[0]}")

hist = model.fit(
    X_train,
    [y_stage_train, y_ah_train],
    epochs = max_epochs,
    batch_size = batch_size,
    validation_split = 0.2, 
    callbacks = [
        cb_timer,
        cb_early_stopping,
        cb_checkpoint,
        lr_scheduler
    ]
)

scores = model.evaluate(X_test, [y_stage_test, y_ah_test], batch_size=batch_size, verbose=False)[1::]

print("\nTEST RESULT\n")

f = open(path.join("history", f"{name}_logs.txt"), "w")
t = sum(cb_timer.logs)
print(f"Total training time: {convert_seconds(t)}")
print(f"Total training time: {convert_seconds(t)}", file=f)
print(f"Total epochs: {len(cb_timer.logs)}\n")
print(f"Total epochs: {len(cb_timer.logs)}\n", file=f)
for threshold in range(1, 10):
    print(f"Threshold 0.{threshold}: {scores[threshold-1]}")
    print(f"Threshold 0.{threshold}: {scores[threshold-1]}", file=f)
f.close()

for key, value in hist.history.items():
    data = np.array(value)
    his_path = path.join("history", f"{name}_{key}_ECG")
    np.save(his_path, data)

print("Saving history done!")
