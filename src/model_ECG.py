from model_functions import *

def create_model_ECG():
    # sleep time steps, 1
    inp = layers.Input(shape=(None, 1))
    
    # downsample
    x = layers.Conv1D(filters=64, kernel_size=5)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.AvgPool1D(pool_size=2)(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    
    for _ in range(4):
        x = ResNetBlock(
            dimension = 1,
            inp = x,
            filters = 64,
            down_sample = True,
            pool = layers.AvgPool1D(pool_size=2)
        )
        x = SEBlock(reduction_ratio=4)(x)

    
    x = MyMultiHeadRelativeAttention(num_heads=16, depth=32, max_relative_position=640)(x) # max relative postion = 5s (128hz),
    
    x = layers.GlobalAvgPool1D()(x)
    out = layers.Dense(1)(x)
    
    model = Model(
        inputs = inp, 
        outputs = out
    )
    
    return model

save_path = path.join("res", "model_ECG.weights.h5")
model = create_model_ECG()
model.compile(
    optimizer = "adam",
    loss = "mse",
    metrics = ["mae", metrics.RootMeanSquaredError(name="rmse")]
)
model.summary()

name = sys.argv[sys.argv.index("id")+1]

max_epochs = 200
batch_size = 1

# callbacks
early_stopping_epoch = 50
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 3,
)
cb_checkpoint = cbk.ModelCheckpoint(
    save_path, 
    save_best_only=True,
    save_weights_only=True,
)
cb_timer = TimingCallback()
lr_scheduler = cbk.ReduceLROnPlateau(
    factor = 0.5,
    min_lr = 0.000001,
    patience = 5,
)

maxlen = 13880

sequences = []
AHI = []

for i in range(1, 26):
    sequences.append(np.load(path.join("patients", f"patients_{i}_ECG.npy")))
    AHI.append(float(open(path.join("patients", f"patients_{i}_AHI.txt"), "r").readline()))

sequences = np.array(pad_sequences(sequences, maxlen=maxlen, value=0, padding="post"))
AHI = np.array(AHI)

X_train, X_test, y_train, y_test = train_test_split(sequences, AHI, test_size=0.2,random_state=np.random.randint(22022009))
print(f"Train size: {X_train.shape[0]} - Test size: {X_test.shape[0]}")

hist = model.fit(
    X_train,
    y_train,
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

scores = model.evaluate(X_test, y_test, batch_size=batch_size)

f = open(path.join("history", f"{name}_logs.txt"), "w")
print(f"MSE: {scores[0]}, MAE: {scores[1]}, RMSE: {scores[2]}")
print(f"MSE: {scores[0]}, MAE: {scores[1]}, RMSE: {scores[2]}", file=f)
print(f"Total epochs: {len(cb_timer.logs)}")
print(f"Total epochs: {len(cb_timer.logs)}", file=f)
t = sum(cb_timer.logs)
print(f"Total training time: {convert_seconds(t)}")
print(f"Total training time: {convert_seconds(t)}", file=f)
f.close()

for key, value in hist.history.items():
    data = np.array(value)
    his_path = path.join("history", f"{name}_{key}_ECG")
    np.save(his_path, data)

print("Saving history done!")
