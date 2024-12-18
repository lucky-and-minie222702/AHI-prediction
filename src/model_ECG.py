from model_functions import *
from data_functions import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

def create_model_ECG(name: str):    
    # 1000, 1 - 10 seconds
    inp = layers.Input(shape=(None, 1))
    conv = layers.Normalization()(inp)
    
    # down sample
    conv = layers.Conv1D(filters=64, kernel_size=7, strides=2, kernel_regularizer=reg.L2())(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("tanh")(conv)
    conv = layers.MaxPool1D(pool_size=3, strides=2)(conv)
    
    # for stage detecting 
    stage_conv = ResNetBlock(1, conv, 64, True)
    stage_conv = ResNetBlock(1, stage_conv, 64)
    stage_conv = ResNetBlock(1, stage_conv, 64)
    
    stage_conv = SEBlock(reduction_ratio=2)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 128, True)
    stage_conv = ResNetBlock(1, stage_conv, 128)
    stage_conv = ResNetBlock(1, stage_conv, 128)
    
    stage_conv = SEBlock(reduction_ratio=4)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 256, True)
    stage_conv = ResNetBlock(1, stage_conv, 256)
    stage_conv = ResNetBlock(1, stage_conv, 256)
    
    stage_conv = SEBlock(reduction_ratio=6)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 512, True)
    stage_conv = ResNetBlock(1, stage_conv, 512)
    stage_conv = ResNetBlock(1, stage_conv, 512)
    
    stage_att = SEBlock(reduction_ratio=8)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 1024, True)
    stage_conv = ResNetBlock(1, stage_conv, 1024)
    stage_conv = ResNetBlock(1, stage_conv, 1024)
    
    stage_att = SEBlock(reduction_ratio=10)(stage_conv)
    
    stage_conv = ResNetBlock(1, stage_conv, 2048, True)
    stage_conv = ResNetBlock(1, stage_conv, 2048)
    stage_conv = ResNetBlock(1, stage_conv, 2048)
    
    stage_att = SEBlock(reduction_ratio=12)(stage_conv)

    stage_flat = layers.GlobalAvgPool1D()(stage_att)
    stage_flat = layers.Flatten()(stage_flat)
    stage_out = layers.Dense(1, activation="sigmoid", name="stage")(stage_flat)
    
    
    # for apnea hyponea detecting
    ah_conv = ResNetBlock(1, conv, 64, True)
    ah_conv = ResNetBlock(1, ah_conv, 64)
    ah_conv = ResNetBlock(1, ah_conv, 64)
    
    ah_conv = SEBlock(reduction_ratio=2)(ah_conv)
    
    ah_conv = ResNetBlock(1, ah_conv, 128, True)
    ah_conv = ResNetBlock(1, ah_conv, 128)
    ah_conv = ResNetBlock(1, ah_conv, 128)
    
    ah_conv = SEBlock(reduction_ratio=4)(ah_conv)

    ah_conv = ResNetBlock(1, ah_conv, 256, True)
    ah_conv = ResNetBlock(1, ah_conv, 256)
    ah_conv = ResNetBlock(1, ah_conv, 256)
    
    ah_conv = SEBlock(reduction_ratio=6)(ah_conv)
    
    ah_conv = ResNetBlock(1, ah_conv, 512, True)
    ah_conv = ResNetBlock(1, ah_conv, 512)
    ah_conv = ResNetBlock(1, ah_conv, 512)
    
    ah_att = SEBlock(reduction_ratio=8)(ah_conv)
    
    ah_conv = ResNetBlock(1, ah_conv, 1024, True)
    ah_conv = ResNetBlock(1, ah_conv, 1024)
    ah_conv = ResNetBlock(1, ah_conv, 1024)
    
    ah_att = SEBlock(reduction_ratio=10)(ah_conv)
    
    ah_conv = ResNetBlock(1, ah_conv, 2048, True)
    ah_conv = ResNetBlock(1, ah_conv, 2048)
    ah_conv = ResNetBlock(1, ah_conv, 2048)
    
    ah_att = SEBlock(reduction_ratio=12)(ah_conv)

    ah_flat = layers.GlobalAvgPool1D()(ah_att)
    ah_flat = layers.Flatten()(ah_flat)
    ah_out = layers.Dense(1, activation="sigmoid", name="ah")(ah_flat)
    
    model = Model(
        inputs = inp,
        outputs = [stage_out, ah_out],
        name = name
    )

    # show_params(model, name)
    model.summary()
        
    return model

save_path = path.join("res", "model_ECG.weights.h5")
model = create_model_ECG("ECG")
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

sequences = np.load(path.join("patients", "merged_ECG.npy"))
annotations  = np.load(path.join("patients", "merged_anns.npy"))
stages = np.load(path.join("patients", "merged_stages.npy"))
train_indices = np.load(path.join("patients", "train_indices.npy"))
test_indices = np.load(path.join("patients", "test_indices.npy"))

X_train = sequences[train_indices]
y_stage_train = stages[train_indices]
y_ah_train = annotations[train_indices]
X_test = sequences[test_indices]
y_stage_test = stages[test_indices]
y_ah_test = annotations[test_indices]

class_weights_stage = compute_class_weight('balanced', classes=np.unique(y_stage_train), y=y_stage_train)
class_weights_ah = compute_class_weight('balanced', classes=np.unique(y_ah_train), y=y_ah_train)

sample_weights_stage = np.array([class_weights_stage[int(label)] for label in y_stage_train])
sample_weights_ah = np.array([class_weights_ah[int(label)] for label in y_ah_train])

sample_weights_dict = {
    "stage": sample_weights_stage,
    "ah": sample_weights_ah,
}

model.compile(
    optimizer = "Adam",
    loss = {
        "stage": "binary_crossentropy",
        "ah": "binary_crossentropy",
    },
    metrics = {
        "stage": [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
        "ah": [metrics.BinaryAccuracy(name = f"threshold_0.{t}", threshold = t/10) for t in range(1, 10)],
    }
)

print(f"\nTrain size: {X_train.shape[0]} - Test size: {X_test.shape[0]}\n")

hist = model.fit(
    X_train,
    {
        "stage": y_stage_train, 
        "ah": y_ah_train
    },
    epochs = max_epochs,
    batch_size = batch_size,
    validation_split = 0.2, 
    sample_weight = sample_weights_dict,
    callbacks = [
        cb_timer,
        cb_early_stopping,
        cb_checkpoint,
        lr_scheduler
    ]
)

scores = model.evaluate(X_test, {"stage": y_stage_test, "ah": y_ah_test}, batch_size=batch_size, return_dict=True)

print("\nSUMMARY\n")

f = open(path.join("history", f"{name}_logs_ECG.txt"), "w")
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

f.close()

for key, value in hist.history.items():
    data = np.array(value)
    his_path = path.join("history", f"{name}_{key}_ECG")
    np.save(his_path, data)

print("Saving history done!")
