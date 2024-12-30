from sympy import sequence
from model_functions import *
from data_functions import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

def create_model_SpO2_ah(name: str):
    inp = layers.Input(shape=(None, None, 1))
    x = layers.Normalization()(inp)
    
    x = layers.TimeDistributed(layers.Conv1D(filters=16, kernel_size=5))(x)
    x = layers.TimeDistributed(layers.MaxPool1D(pool_size=3, strides=2))(x)
    x = layers.TimeDistributed(layers.GlobalAvgPool1D())(x)
    
    x = ResNetBlock(1, x, 64, 3, True)
    x = ResNetBlock(1, x, 64, 3)
    
    x = ResNetBlock(1, x, 128, 3, True)
    x = ResNetBlock(1, x, 128, 3)

    x = ResNetBlock(1, x, 256, 3, True)
    x = ResNetBlock(1, x, 256, 3)
    
    x = SEBlock(reduction_ratio=2)(x)
    
    x = layers.LSTM(64)(x)
    
    x = layers.GlobalAvgPool1D()(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    out = layers.Dense(1)(x)

    model = Model(
        inputs = inp,
        outputs = out,
        name = name
    )

    show_params(model, name)
    # model.summary()
        
    return model

model = create_model_SpO2_ah("SpO2_ah")
name = sys.argv[sys.argv.index("id")+1]
save_path = path.join("res", f"model_SpO2_ah{name if name != '1' else ''}.weights.h5")

model.compile(
    optimizer = "Adam",
    loss =  "mse",
    # metrics = ["accuracy"]
    # metrics = [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
    # metrics = [metrics.Precision(name = f"precision_threshold_0.{t}", threshold = t/10) for t in range(1, 10)] + 
    #           [metrics.Recall(name = f"precision_threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
)

max_epochs = 1 if "test_save" in sys.argv else 500
batch_size = 64
if "batch_size" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("batch_size")+1])

majority_weight = 1.0
if "mw" in sys.argv:
    majority_weight = float(sys.argv[sys.argv.index("mw")+1])

# callbacks
early_stopping_epoch = 200
if "ese" in sys.argv:
    early_stopping_epoch = int(sys.argv[sys.argv.index("ese")+1])
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
    patience = 7,
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

sequences = []
annotations = []
maxlen = 0
for i in range(1, 26):
    seq_SpO2 = np.load(path.join("patients", f"patients_{i}_SpO2.npy"))
    ann = float(open(path.join("patients", f"patients_{i}_AHI.txt")).readlines()[-1])
    maxlen = max(maxlen, len(seq_SpO2))
    sequences.append(seq_SpO2)
    annotations.append(ann)
    
sequences = pad_sequences(sequences, maxlen=maxlen)
sequences = np.array([divide_signal([x], win_size=30, step_size=5)[0] for x in sequences])
annotations = np.array(annotations)

sequences = np.vstack(
    [sequences, sequences + np.random.normal(0.0, 0.01, sequences.shape)]
)

annotations = np.concatenate([
    annotations, annotations
])
annotations /= 10

# threshold = 1.5

# annotations = np.array([1 if x >= threshold else 0 for x in annotations])

if "train" in sys.argv:
    indices = np.arange(len(annotations))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=random.randint(69, 69696969))
    np.save(path.join("patients", "train_indices_SpO2_ah"), train_indices)
    np.save(path.join("patients", "test_indices_SpO2_ah"), test_indices)
        
    X_train = sequences[train_indices]
    y_train = annotations[train_indices]
    X_test = sequences[test_indices]
    y_test = annotations[test_indices]

    print("Dataset:")
    # print(f"Train set: [0]: {np.count_nonzero(y_train == 0)}  |  [1]: {np.count_nonzero(y_train == 1)}")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random.randint(69, 69696969))

    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weight = dict(enumerate(class_weights))
    # sample_weights = np.array([class_weights[int(label)] for label in y_train])

    print(f"\nTrain size: {X_train.shape[0]}")

    hist = model.fit(
        X_train,
        y_train,
        epochs = max_epochs,
        batch_size = batch_size,
        validation_data = (X_val, y_val),
        # class_weight = class_weight,
        callbacks = [
            cb_timer,
            cb_early_stopping,
            cb_checkpoint,
            lr_scheduler
        ]
    )

test_indices = np.load(path.join("patients", "test_indices_SpO2_ah.npy"))
X_test = sequences[test_indices]
y_test = annotations[test_indices]

model.load_weights(save_path)

# class_weights = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)
# class_weight = dict(enumerate(class_weights))
# sample_weights = np.array([class_weights[int(label)] for label in y_test])

# print(f"Test set: [0]: {np.count_nonzero(y_test == 0)}  |  [1]: {np.count_nonzero(y_test == 1)}")

scores = model.evaluate(
    X_test, 
    y_test,
    # sample_weight = sample_weights,
    batch_size = batch_size, 
    return_dict = True
)

y_test = annotations[test_indices]
# if "balance" in sys.argv:
#     y_test = y_test[combined_balance]

print("\nSUMMARY\n")

f = open(path.join("history", f"{name}_logs_SpO2_ah.txt"), "w")
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

raw_pred = model.predict(X_test, verbose=False, batch_size=batch_size).squeeze() * 10

# for d in range(1, 10):
#     threshold = d / 10
#     print(f"Threshold 0.{d}")
#     print(f"Threshold 0.{d}", file=f)
#     arr = np.array([np.squeeze(x) for x in raw_pred])
#     pred =  np.where(arr % 1 >= threshold, np.ceil(arr), np.floor(arr))
#     cm = confusion_matrix(y_test, pred)
#     print("Confusion matrix:\n", cm)
#     print("Confusion matrix:\n", cm, file=f)
#     print(calc_cm(cm))
    # print(calc_cm(cm), file=f)

print("Real - Prediction:")
print("Real - Predicti/on:", file=f)
for i, ans in enumerate(y_test):
    print(ans * 10, raw_pred[i])
    print(ans * 10, raw_pred[i], file=f)
    
    

f.close()

if "train" in sys.argv:
    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{name}_{key}_SpO2_ah")
        np.save(his_path, data)

    print("Saving history done!")
