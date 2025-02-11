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
    conv_rpa = layers.BatchNormalization()(norm_inp_rpa)
    conv_rpa = ResNetBlock(1, norm_inp_rpa, 64, 3, change_sample=True)
    conv_rpa = ResNetBlock(1, conv_rpa, 64, 3)
    
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3, change_sample=True)
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3)
    conv_rpa = ResNetBlock(1, conv_rpa, 128, 3)
    
    inp_rri = layers.Input(shape=(None, 1))
    norm_inp_rri = layers.Normalization()(inp_rri)
    conv_rri = ResNetBlock(1, norm_inp_rri, 64, 3, change_sample=True)
    conv_rri = ResNetBlock(1, conv_rri, 64, 3)
    
    conv_rri = ResNetBlock(1, conv_rri, 128, 3, change_sample=True)
    conv_rri = ResNetBlock(1, conv_rri, 128, 3)
    conv_rri = ResNetBlock(1, conv_rri, 128, 3)
    
    r_peak_features = layers.Concatenate(axis=-2)([conv_rpa, conv_rri])
    r_peak_features = ResNetBlock(1, r_peak_features, 256, 3, change_sample=True)
    r_peak_features = ResNetBlock(1, r_peak_features, 256, 3)
    r_peak_features = SEBlock()(r_peak_features)
    
    inp = layers.Input(shape=(None, 1))
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
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3) 
    
    conv = layers.Attention(use_scale=True)([conv, r_peak_features, r_peak_features])
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    
    # bottle-neck
    conv_bn = layers.Conv1D(filters=256, kernel_size=1, strides=1, padding="same")(conv)
    conv_bn = layers.BatchNormalization()(conv_bn)
    conv_bn = layers.Activation("relu")(conv_bn)
    
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(conv_bn)
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn)
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn)
    
    # restore    
    conv_r = layers.Conv1D(filters=512, kernel_size=1, padding="same")(rnn)
    conv_r = layers.BatchNormalization()(conv_r)
    conv_r = layers.Activation("relu")(conv_r)
    conv_r = layers.Add()([conv, conv_r])  # residual connection
    conv_r = layers.Activation("relu")(conv_r)
    conv_r = ResNetBlock(1, conv_r, 512, 1)
    
    se = SEBlock()(conv_r)
    
    fc = layers.GlobalAvgPool1D()(se)
    fc = layers.Dense(512)(fc)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Activation("relu")(fc)
    
    # single output
    out_s = layers.Dense(512)(fc)
    out_s = layers.BatchNormalization()(out_s)
    out_s = layers.Activation("relu")(out_s)
    out_s = layers.Dense(1, activation="sigmoid", name="single")(out_s)
    
    # preserved input shape (shallow features extract)
    pis = ResNetBlock(1, ds_conv, 64, 9, change_sample=5)
    pis = ResNetBlock(1, pis, 64, 7)
    pis = ResNetBlock(1, pis, 128, 5, change_sample=5)
    pis = ResNetBlock(1, pis, 128, 3)
    pis = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(pis)
    pis = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(pis)
    pis = ResNetBlock(1, pis, 512, 1)  # match channel
    pis = layers.Cropping1D(cropping=(10, 10))(pis)  # match length
    
    # full-segment output
    out = layers.Dot(axes=-1)([pis, fc])
    out = layers.Activation("sigmoid", name="full")(out)
    
    
    model = Model(inputs=[inp, inp_rpa, inp_rri], outputs=[out, out_s])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = {
            "full": "binary_crossentropy",
            "single": "binary_crossentropy",
        },
        metrics = {
            "single": [metrics.BinaryAccuracy(name = f"t=0.{t}", threshold = t/10) for t in range(1, 10)],
        }
    )

    return model

model = create_model()
show_params(model, "ecg_ah")
weights_path = path.join("weights", ".weights.h5")
model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 128
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 50,
    patience = 10,
)
cb_checkpoint = cbk.ModelCheckpoint(
    weights_path, 
    save_best_only = True,
    save_weights_only = True,
    monitor = "val_full_loss",
    mode = "min",
)

cb_lr = cbk.ReduceLROnPlateau(monitor='val_full_loss', factor=0.2, patience=10, min_lr=0.00001)

print("\nTRAINING\n")

# clean_method = ['pantompkins1985', 'hamilton2002', 'elgendi2010', 'vg']
# total_test_indices = []

ecgs = np.load(path.join("gen_data", "merged_ecg.npy"))
full_labels = np.load(path.join("data", "merged_label.npy"))
rpa = np.load(path.join("gen_data", "merged_rpa.npy"))
rri = np.load(path.join("dgen_ata", "merged_rri.npy"))

mean_labels = np.mean(full_labels, axis=-1)
single_labels = np.round(mean_labels)
sample_weights = np.ones(shape=single_labels.shape)
sample_weights += mean_labels

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

show_res(single_labels[test_indices], single_preds)

np.save(path.join("history", "ecg_test_result"), np.vstack([single_labels[test_indices], single_preds]))