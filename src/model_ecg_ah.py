from numpy import ndim
from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

show_gpus()
        
        
def create_model():
    inp_rpa = layers.Input(shape=(None, 1))
    norm_inp_rpa = layers.Normalization()(inp_rpa)
    conv_rpa = ResNetBlock(1, norm_inp_rpa, 64, 3, change_sample=True)
    conv_rpa = ResNetBlock(1, conv_rpa, 64, 3)
    
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3, change_sample=True)
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3)
    
    inp_rri = layers.Input(shape=(None, 1))
    norm_inp_rri = layers.Normalization()(inp_rri)
    conv_rri = ResNetBlock(1, norm_inp_rri, 64, 3, change_sample=True)
    conv_rri = ResNetBlock(1, conv_rri, 64, 3)
    
    conv_rri = ResNetBlock(1, conv_rri, 128, 3, change_sample=True)
    conv_rri = ResNetBlock(1, conv_rri, 128, 3)
    
    r_peak_features = layers.Concatenate(axis=-2)([conv_rpa, conv_rri])
    r_peak_features = ResNetBlock(1, r_peak_features, 256, 3, change_sample=True)
    r_peak_features = ResNetBlock(1, r_peak_features, 256, 3)
    r_peak_features = SEBlock()(r_peak_features)
    
    inp = layers.Input(shape=(3100, 1))  # 30s
    norm_inp = layers.Normalization()(inp)
    
    # down_sample
    ds_conv = layers.Conv1D(filters=64, kernel_size=11, strides=2, padding="same")(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=2)(ds_conv)
    
    # deep
    conv = ResNetBlock(1, ds_conv, 64, 9)
    conv = ResNetBlock(1, conv, 64, 7)
    conv = ResNetBlock(1, conv, 64, 5)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3) 
    
    r_peak_att = layers.Attention(use_scale=True)([conv, r_peak_features, r_peak_features])
    
    # bottle-neck lstm
    conv_bn1 = layers.Conv1D(filters=64, kernel_size=1, padding="same")(r_peak_att)
    conv_bn1 = layers.BatchNormalization()(conv_bn1)
    conv_bn1 = layers.Activation("relu")(conv_bn1)
    
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(conv_bn1)
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn)
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn)
    
    # restore    
    conv_r1 = layers.Add()([conv, rnn])  # residual connection
    conv_r1 = layers.Activation("relu")(conv_r1)
    
    conv_r1 = ResNetBlock(1, conv_r1, 512, 3, change_sample=True)
    conv_r1 = ResNetBlock(1, conv_r1, 512, 3)
    conv_r1 = ResNetBlock(1, conv_r1, 512, 3)
    
    # bottle-neck attention
    conv_bn2 = layers.Conv1D(filters=128, kernel_size=1, padding="same")(conv_r1)
    conv_bn2 = layers.BatchNormalization()(conv_bn2)
    conv_bn2 = layers.Activation("relu")(conv_bn2)
    
    att = MyAtt(depth=64, num_heads=16, seq_len=97)(conv_bn2, conv_bn2, conv_bn2)
    full = layers.Conv1D(filters=512, kernel_size=1, padding="same")(att)
    full = layers.BatchNormalization()(full)
    full = layers.Activation("relu")(full)
    full = layers.Add()([full, conv_r1])  # residual connection
    full = layers.Activation("relu")(full)
    
    conv_r2 = ResNetBlock(1, full, 1024, 3, change_sample=True)
    conv_r2 = ResNetBlock(1, conv_r2, 1024, 3)
    conv_r2 = ResNetBlock(1, conv_r2, 1024, 3)
    
    se1 = SEBlock()(conv_r2)
    se2 = SEBlock()(conv_r2)
    
    # single second
    out_s = layers.GlobalAvgPool1D()(se1)
    out_s = layers.Dense(1024)(out_s)
    out_s = layers.BatchNormalization()(out_s)
    out_s = layers.Activation("relu")(out_s)
    out_s = layers.Dense(1024)(out_s)
    out_s = layers.BatchNormalization()(out_s)
    out_s = layers.Activation("relu")(out_s)
    out_s = layers.Dense(1, activation="sigmoid", name="single")(out_s)
    
    # full segment
    out_f = layers.GlobalAvgPool1D()(se2)
    out_f = layers.Dense(1024)(out_f)
    out_f = layers.BatchNormalization()(out_f)
    out_f = layers.Activation("relu")(out_f)
    out_f = layers.Dense(1024)(out_f)
    out_f = layers.BatchNormalization()(out_f)
    out_f = layers.Activation("relu")(out_f)
    out_f = layers.Dense(1, activation="sigmoid", name="full")(out_f)

    
    model = Model(inputs=[inp, inp_rpa, inp_rri], outputs=[out_f, out_s])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = {
            "full": "binary_crossentropy",
            "single": "binary_crossentropy",
        },
        metrics = {
            "full": [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
            "single": [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
        }
    )

    return model

model = create_model()
show_params(model, "ecg_ah")

weights_path = path.join("weights", ".weights.h5")
model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 64
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 100,
    patience = 10,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
    monitor = "val_single_loss",
    mode = "min",
)

cb_lr = cbk.ReduceLROnPlateau(monitor='val_single_loss', mode="min", factor=0.2, patience=10, min_lr=0.00001)

print("\nTRAINING\n")

# clean_method = ['pantompkins1985', 'hamilton2002', 'elgendi2010', 'vg']
# total_test_indices = []

ecgs = np.load(path.join("gen_data", "merged_ecg.npy"))
full_labels = np.load(path.join("gen_data", "merged_label.npy"))
rpa = np.load(path.join("gen_data", "merged_rpa.npy"))
rri = np.load(path.join("gen_data", "merged_rri.npy"))

mean_labels = np.mean(full_labels, axis=-1)
full_labels = np.round(mean_labels)
single_labels = np.array([l[15] for l in full_labels])

total_samples = len(ecgs)
indices = np.arange(total_samples)
indices = np.random.permutation(indices)
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.15)
test_size = total_samples - train_size - val_size

train_indices = indices[:train_size:]
test_indices = indices[train_size:train_size+test_size:]
val_indices = indices[train_size+test_size::]

print(f"\nTrain - Test - Val: {train_size} - {test_size} - {val_size}")
class_counts = np.unique(single_labels[train_indices], return_counts=True)[1]
print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}\n")

sample_weights = np.ones(shape=mean_labels.shape)
sample_weights += mean_labels

model.load_weights(weights_path)
model.fit(
    [ecgs[train_indices], rpa[train_indices], rri[train_indices]],
    [full_labels[train_indices], single_labels[train_indices]],
    epochs = epochs,
    validation_data = (
        [ecgs[val_indices], rpa[val_indices], rri[val_indices]],
        [full_labels[val_indices], single_labels[val_indices]]
    ),
    batch_size = batch_size,
    callbacks = [cb_early_stopping, cb_checkpoint, cb_lr],
    sample_weight = sample_weights[train_indices],
)

class_counts = np.unique(single_labels[test_indices], return_counts=True)[1]
print("\nTESTING\n")
print(f"Class 0: {class_counts[0]} - Class 1: {class_counts[1]}")
raw_preds = model.predict([ecgs[test_indices], rpa[test_indices], rri[test_indices]], batch_size=batch_size)
full_preds = raw_preds[0]
single_preds = raw_preds[1]

np.save(path.join("history", "ecg_test_result"), np.vstack([single_labels[test_indices], single_preds]))

print("\nClassification on single second result:\n")
show_res(single_labels[test_indices], single_preds)
print("\nClassification on full segment result:\n")
show_res(full_labels[test_indices], full_preds)
