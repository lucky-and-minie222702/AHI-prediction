from model_functions import *
from data_functions import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

def create_model_ECG_ah(name: str):    
    # 500, 5 seconds
    inp = layers.Input(shape=(None, 1))
    norm_inp = layers.Normalization()(inp)
    
    conv = layers.Conv1D(filters=64, kernel_size=7, strides=2)(norm_inp)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.MaxPool1D(pool_size=3, strides=2)(conv)

    conv = ResNetBlock(1, conv, 64, 11)
    conv = ResNetBlock(1, conv, 64, 11)
    conv = ResNetBlock(1, conv, 64, 11)
    
    conv = ResNetBlock(1, conv, 128, 9, True)
    conv = ResNetBlock(1, conv, 128, 9)
    conv = ResNetBlock(1, conv, 128, 9)
    conv = ResNetBlock(1, conv, 128, 9)
    
    conv = ResNetBlock(1, conv, 256, 7, True)
    conv = ResNetBlock(1, conv, 256, 7)
    conv = ResNetBlock(1, conv, 256, 7)
    conv = ResNetBlock(1, conv, 256, 7)
    conv = ResNetBlock(1, conv, 256, 7)
    conv = ResNetBlock(1, conv, 256, 7)
    
    conv = ResNetBlock(1, conv, 512, 5, True)
    conv = ResNetBlock(1, conv, 512, 5)
    conv = ResNetBlock(1, conv, 512, 5)
    
    conv = ResNetBlock(1, conv, 1024, 3, True)
    
    conv = MyMultiHeadRelativeAttention(depth=32, num_heads=32, max_relative_position=16)(conv)
    
    conv = SEBlock(reduction_ratio=4)(conv)

    flat = layers.GlobalAvgPool1D()(conv)
    flat = layers.Flatten()(flat)
    out = layers.Dense(2, activation="softmax")(flat)
    
    model = Model(
        inputs = inp,
        outputs = out,
        name = name
    )

    show_params(model, name)
    # model.summary()
        
    return model

save_path = path.join("res", "model_ECG_ah.weights.h5")
model = create_model_ECG_ah("ECG_ah")
name = sys.argv[sys.argv.index("id")+1]

model.compile(
    optimizer = "Adam",
    loss =  "binary_crossentropy",
    metrics = ["accuracy"]
    # metrics = [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
    # metrics = [metrics.Precision(name = f"precision_threshold_0.{t}", threshold = t/10) for t in range(1, 10)] + 
    #           [metrics.Recall(name = f"precision_threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
)

max_epochs = 1 if "test_save" in sys.argv else 200
batch_size = 64
if "batch_size" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("batch_size")+1])

majority_weight = 1.0
if "mw" in sys.argv:
    majority_weight = float(sys.argv[sys.argv.index("mw")+1])

# callbacks
early_stopping_epoch = 50
if "ese" in sys.argv:
    early_stopping_epoch = int(sys.argv[sys.argv.index("ese")+1])
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 5,
)
cb_checkpoint = cbk.ModelCheckpoint(
    save_path, 
    save_best_only = True,
    save_weights_only = True,
)
cb_timer = TimingCallback()
lr_scheduler = cbk.ReduceLROnPlateau(
    factor = 0.5,
    min_lr = 0.00001,
    patience = 5,
)

sequences = np.load(path.join("patients", "merged_ECG.npy"))
annotations  = np.load(path.join("patients", "merged_anns.npy"))
annotations = np.concatenate([
    annotations, annotations,
])
print(len(sequences), len(annotations))
exit()

if "train" in sys.argv:
    indices = np.arange(len(annotations))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(69696969))
    np.save(path.join("patients", "train_indices_ECG_ah"), train_indices)
    np.save(path.join("patients", "test_indices_ECG_ah"), test_indices)
        
    X_train = sequences[train_indices]
    y_train = annotations[train_indices]
    X_test = sequences[test_indices]
    y_test = annotations[test_indices]

    if "balance" in sys.argv:
        # Train set
        balance = balancing_data(y_train, majority_weight)
        combined_balance = np.unique(balance)

        X_train = X_train[combined_balance]
        y_train = y_train[combined_balance]

    print("Dataset:")
    print(f"Train set: [0]: {np.count_nonzero(y_train == 0)}  |  [1]: {np.count_nonzero(y_train == 1)}")
    print(f"Test set: [0]: {np.count_nonzero(y_test == 0)}  |  [1]: {np.count_nonzero(y_test == 1)}")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=np.random.randint(69696969))

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(class_weights))
    sample_weights = np.array([class_weights[int(label)] for label in y_train])

    print(f"\nTrain size: {X_train.shape[0]} - Test size: {X_test.shape[0]}\n")

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)

    hist = model.fit(
        X_train,
        y_train,
        epochs = max_epochs,
        batch_size = batch_size,
        validation_data = (X_val, y_val),
        class_weight = class_weight,
        callbacks = [
            cb_timer,
            cb_early_stopping,
            cb_checkpoint,
            lr_scheduler
        ]
    )

test_indices = np.load(path.join("patients", "test_indices_ECG_ah.npy"))
X_test = sequences[test_indices]
y_test = annotations[test_indices]

if "balance" in sys.argv:
    # Test set
    balance = balancing_data(y_test, majority_weight)
    combined_balance = np.unique(balance)

    X_test = X_test[combined_balance]
    y_test = y_test[combined_balance]

model.load_weights(save_path)

class_weights = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)
class_weight = dict(enumerate(class_weights))
sample_weights = np.array([class_weights[int(label)] for label in y_test])

y_test = to_categorical(y_test, num_classes=2)

scores = model.evaluate(
    X_test, 
    y_test,
    sample_weight = sample_weights,
    batch_size = batch_size, 
    return_dict = True
)

y_test = annotations[test_indices]
if "balance" in sys.argv:
    y_test = y_test[combined_balance]

print("\nSUMMARY\n")

f = open(path.join("history", f"{name}_logs_ECG_ah.txt"), "w")
t = sum(cb_timer.logs)
print(f"Total training time: {convert_seconds(t)}")
print(f"Total training time: {convert_seconds(t)}", file=f)
print(f"Total epochs: {len(cb_timer.logs)}\n")
print(f"Total epochs: {len(cb_timer.logs)}\n", file=f)

print("\nTEST RESULT\n")
print("\nTEST RESULT\n", file=f)

for metric, score in scores.items():
    print(f"{metric}: {score}")
    print(f"{metric}: {score}", file=f)

pred = model.predict(X_test, verbose=False, batch_size=batch_size)
arr = np.array([np.squeeze(x) for x in pred])
pred = np.array([np.argmax(x) for x in arr])
cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n", cm)
print("Confusion matrix:\n", cm, file=f)
print(calc_cm(cm))
print(calc_cm(cm), file=f)

f.close()

if "train" in sys.argv:
    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{name}_{key}_ECG_ah")
        np.save(his_path, data)

    print("Saving history done!")
