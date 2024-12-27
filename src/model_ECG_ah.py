from cv2 import threshold
from model_functions import *
from data_functions import *
from sklearn.utils.class_weight import compute_class_weight

def create_model_ECG_ah(name: str):    
    rri_inp = layers.Input(shape=(None, 1))
    rri_conv = layers.Normalization()(rri_inp)
    rri_conv = layers.Conv1D(filters=16, kernel_size=1)(rri_conv)
    rri_conv = layers.BatchNormalization()(rri_conv)
    rri_conv = layers.Activation("relu")(rri_conv)
    rri_conv = layers.GlobalAvgPool1D()(rri_conv)
    rri_conv = layers.Flatten()(rri_conv)
    
    rpa_inp = layers.Input(shape=(None, 1))
    rpa_conv = layers.Normalization()(rpa_inp)
    rpa_conv = layers.Conv1D(filters=16, kernel_size=1)(rpa_conv)
    rpa_conv = layers.BatchNormalization()(rpa_conv)
    rpa_conv = layers.Activation("relu")(rpa_conv)
    rpa_conv = layers.GlobalAvgPool1D()(rpa_conv)
    rpa_conv = layers.Flatten()(rpa_conv)
    
    f_flat = layers.concatenate([rri_conv, rpa_conv])
    f_flat = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(f_flat)
    
    # 500, 5 seconds
    inp = layers.Input(shape=(None, 1))  
    conv = layers.Normalization()(inp)

    conv = ResNetBlock(1, conv, 64, 3, True)
    conv = ResNetBlock(1, conv, 64, 3)
    
    conv = ResNetBlock(1, conv, 128, 3, True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    
    conv = ResNetBlock(1, conv, 512, 3, True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    
    conv = ResNetBlock(1, conv, 1024, 3, True)
    conv = ResNetBlock(1, conv, 1024, 3)
    
    conv = SEBlock(reduction_ratio=2)(conv)

    conv = layers.GlobalAvgPool1D()(conv)
    
    conv = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(conv)
    
    att = layers.MultiHeadAttention(num_heads=8, key_dim=64, value_dim=64)(conv, f_flat, f_flat)

    flat = layers.Flatten()(att)

    flat = layers.Flatten()(flat)
    out = layers.Dense(1, activation="sigmoid")(flat)
    
    model = Model(
        inputs = [inp, rri_inp, rpa_inp],
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
    # metrics = ["accuracy"]
    metrics = [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
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

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=np.random.randint(69696969))
    X_val_rri, X_val_rpa = calc_ecg(X_val)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(class_weights))
    sample_weights = np.array([class_weights[int(label)] for label in y_train])

    X_train_rri, X_train_rpa = calc_ecg(X_train)

    print(f"\nTrain size: {X_train.shape[0]}")

    # y_train = to_categorical(y_train, num_classes=2)
    # y_test = to_categorical(y_test, num_classes=2)
    # y_val = to_categorical(y_val, num_classes=2)

    hist = model.fit(
        [X_train, X_train_rri, X_train_rpa],
        y_train,
        epochs = max_epochs,
        batch_size = batch_size,
        validation_data = ([X_val, X_val_rri, X_val_rpa], y_val),
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

X_test_rri, X_test_rpa = calc_ecg(X_test)

scores = model.evaluate(
    [X_test, X_test_rri, X_test_rpa], 
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

raw_pred = model.predict([X_test, X_test_rri, X_test_rpa], verbose=False, batch_size=batch_size)

for d in range(1, 10):
    threshold = d / 10
    print(f"Threshold 0.{d}")
    print(f"Threshold 0.{d}", file=f)
    arr = np.array([np.squeeze(x) for x in raw_pred])
    pred =  np.where(arr % 1 >= threshold, np.ceil(arr), np.floor(arr))
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
