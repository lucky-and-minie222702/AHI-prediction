from model_functions import *

def create_model_ECG():
    segment_input = layers.Input(shape=(640, 1))
    segment_norm = layers.Normalization()(segment_input)
    segment_conv = ResNetBlock(1, segment_norm, 64, True)
    segment_conv = ResNetBlock(1, segment_conv, 64, True)
    segment_conv = ResNetBlock(1, segment_conv, 64, True)
    segment_att = MyMultiHeadRelativeAttention(depth=64, num_heads=16, max_relative_position=32)(segment_conv)
    segment_model = Model(segment_input, segment_att)
    
    ECG_inp = layers.Input(shape=(None, None, 1))
    segment_outputs = layers.TimeDistributed(segment_model)(ECG_inp)
    aggregated_output1 = layers.GlobalAvgPool2D()(segment_outputs)
    conv = layers.Reshape((list(aggregated_output1.shape[1::]) + [1]))(aggregated_output1)
    conv = ResNetBlock(1, conv, 64, True)
    conv = ResNetBlock(1, conv, 64, True)
    conv = ResNetBlock(1, conv, 64, True)
    att = SEBlock(reduction_ratio=4)(conv)
    aggregated_output2 = layers.GlobalAvgPool1D()(att)
    final_output = layers.Dense(3, activation = "softmax")(aggregated_output2)
    
    
    model = Model(
        inputs = ECG_inp, 
        outputs = final_output,
    )
    
    return model

save_path = path.join("res", "model_ECG.weights.h5")
model = create_model_ECG()
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
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
AHIs = []

for i in range(1, 26):
    sequences.append(np.load(path.join("patients", f"patients_{i}_ECG.npy")))
    AHIs.append(round(float(open(path.join("patients", f"patients_{i}_AHI.txt"), "r").readline())))

sequences = np.array(pad_sequences(sequences, maxlen=maxlen, value=0, padding="post"))
med = np.median(np.array(AHIs))
AHIs = to_categorical(np.array([map_AHI(n) for n in AHIs]), num_classes=3)

# Augmentation
sequences = np.vstack([
    sequences, sequences + np.random.normal(0.0, 0.003, sequences.shape)
])
sequences = np.vstack([
    sequences, add_baseline_wander(sequences)
])

AHIs = np.vstack([
    AHIs, AHIs
])
AHIs = np.vstack([
    AHIs, AHIs
])

segments = divide_signal(sequences, 640)

X_train, X_test, y_train, y_test = train_test_split(segments, AHIs, test_size=0.2,random_state=np.random.randint(22022009))
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

print("\nTEST RESULT\n")

f = open(path.join("history", f"{name}_logs.txt"), "w")
print(f"Loss: {scores[0]}, Metrics: {scores[1]}")
print(f"Loss: {scores[0]}, Metrics: {scores[1]}", file=f)
print(model.metrics_names)
print(model.metrics_names, file=f)
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
