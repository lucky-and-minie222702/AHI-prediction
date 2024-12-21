from model_functions import *
from data_functions import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

def create_model_SpO2(name: str):    
    # min mean median std var
    inp = layers.Input(shape=(10, ))
    e_x = layers.Normalization()(inp)
    e_x = layers.Dense(6, activation="relu")(e_x)
    e_x = layers.Dense(2, activation="relu")(e_x)
    d_x = layers.Dense(6, activation="relu")(e_x)
    out = layers.Dense(10, activation="sigmoid")(d_x)
    
    model = Model(
        inputs = inp,
        outputs = out,
        name = name + "_autoencoder"
    )

    # show_params(model, name)
    model.summary()

    return model

save_path = path.join("res", "model_SpO2.weights.h5")
model = create_model_SpO2("SpO2")
name = sys.argv[sys.argv.index("id")+1]

model.compile(
    optimizer = "Adam",
    loss =  "mse",
    # metrics = [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
    # metrics = [metrics.Precision(name = f"precision_threshold_0.{t}", threshold = t/10) for t in range(1, 10)] + 
    #           [metrics.Recall(name = f"precision_threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
)

max_epochs = 100
batch_size = 64
if "batch_size" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("batch_size")+1])

majority_weight = 1.0
if "mw" in sys.argv:
    majority_weight = float(sys.argv[sys.argv.index("mw")+1])

# callbacks
early_stopping_epoch = 40
if "ese" in sys.argv:
    early_stopping_epoch = int(sys.argv[sys.argv.index("ese")+1])
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = early_stopping_epoch,
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

sequences = np.load(path.join("patients", "merged_SpO2.npy"))
annotations  = np.load(path.join("patients", "merged_anns.npy"))
annotations = np.concatenate([
    annotations, annotations
])

# filter
idx = np.array([
    i for i, val in enumerate(sequences) if np.min(val) >= 0.7
])
sequences = sequences[idx]
annotations = annotations[idx]

if "train" in sys.argv:
    indices = np.arange(len(annotations))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(69696969))
    np.save(path.join("patients", "train_indices_SpO2"), train_indices)
    np.save(path.join("patients", "test_indices_SpO2"), test_indices)
        
    X_train = sequences[train_indices]
    y_train = annotations[train_indices]
    X_train = X_train[y_train == 0]

    X_train, X_val = train_test_split(X_train, test_size=0.2,random_state=np.random.randint(69696969))

    print(f"\nTrain size: {X_train.shape[0]} - Test size: {len(test_indices)}\n")

    hist = model.fit(
        X_train,
        X_train,
        epochs = max_epochs,
        batch_size = batch_size,
        validation_data = (X_val, X_val),
        callbacks = [
            cb_timer,
            cb_early_stopping,
            cb_checkpoint,
            lr_scheduler
        ]
    )

test_indices = np.load(path.join("patients", "test_indices_SpO2.npy"))
X_test = sequences[test_indices]
y_test = annotations[test_indices]
model.load_weights(save_path)

print("\nSUMMARY\n")

f = open(path.join("history", f"{name}_logs_SpO2.txt"), "w")
t = sum(cb_timer.logs)
print(f"Total training time: {convert_seconds(t)}")
print(f"Total training time: {convert_seconds(t)}", file=f)
print(f"Total epochs: {len(cb_timer.logs)}\n")
print(f"Total epochs: {len(cb_timer.logs)}\n", file=f)

print("\nTEST RESULT\n")
print("\nTEST RESULT\n", file=f)

reconstructed_data = model.predict(X_test)
reconstruction_error = np.mean(np.power(X_test - reconstructed_data, 2), axis=1)
threshold = np.percentile(reconstruction_error, 80)
pred = (reconstruction_error > threshold).astype(int) 

cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n", cm)
print("Confusion matrix:\n", cm, file=f)
print(calc_cm(cm))
print(calc_cm(cm), file=f)

# for i in range(1, 10):    
#     threshold = i / 10
#     print(f"Threshold 0.{i}")
#     print(f"Threshold 0.{i}", file=f)
#     pred = model.predict(X_test, verbose=False, batch_size=batch_size)
#     arr = np.array([np.squeeze(x) for x in pred])
#     pred =  np.where(arr % 1 >= threshold, np.ceil(arr), np.floor(arr))
#     cm = confusion_matrix(y_test, pred)
#     print("Confusion matrix:\n", cm)
#     print("Confusion matrix:\n", cm, file=f)
#     print(calc_cm(cm))
#     print(calc_cm(cm), file=f)

f.close()

if "train" in sys.argv:
    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{name}_{key}_SpO2")
        np.save(his_path, data)

    print("Saving history done!")
