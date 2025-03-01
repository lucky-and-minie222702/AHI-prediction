from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler

show_gpus()

def augment_ecg(signal):
    signal = time_warp(signal, sigma=0.075)
    signal = add_noise(signal, noise_std=0.05)
    signal = time_shift(signal, shift_max=20)
    signal *= (np.random.randint(90, 110) / 100)
    return signal

def data_generator(X, y, batch_size):
    rpa0, rri0 = calc_ecg(X[np.where(y == 0)[0]], 100, 30)
    rpa1, rri1 = calc_ecg(X[np.where(y == 1)[0]], 100, 30)
    def generator():
        while True:
            indices0 = np.arange(len(rpa0))
            indices1 = np.arange(len(rpa1))
            np.random.shuffle(indices0)
            np.random.shuffle(indices1)
            for start in range(0, len(rpa0), batch_size):
                end = min(start + batch_size, len(rpa0))
                if (end - start + 1) % 2 != 0:
                    end -= 1
                batch_indices0 = indices0[start:end]
                batch_indices1 = indices1[start:end]
                
                rpa0_batch = rpa0[batch_indices0]
                rri0_batch = rri0[batch_indices0]
                y0_batch = np.full((end-start+1,), 0)
                
                rpa1_batch = rpa1[batch_indices0]
                rri1_batch = rri1[batch_indices1]
                y1_batch = np.full((end-start+1,), 1)
                
                if np.random.rand() >= 0.5:
                    rpa0_batch, rpa1_batch  = rpa1_batch, rpa0_batch
                    rri0_batch, rri1_batch  = rri1_batch, rri0_batch
                    y0_batch, y1_batch  = y1_batch, y0_batch
                    
                rpa = np.concatenate([rpa0_batch, rpa1_batch], axis=0)
                rri = np.concatenate([rri0_batch, rri1_batch], axis=0)
                y = np.concatenate([y0_batch, y1_batch], axis=0)

                yield tuple(rpa, rri), y 
    
    return tf.data.Dataset.from_generator(generator, output_signature=(
        [tf.TensorSpec(shape=(None, *rpa0.shape[1:]), dtype=tf.float32), tf.TensorSpec(shape=(None, *rri0.shape[1:]), dtype=tf.float32)],
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    ))


def contrastive_loss(temperature):
    def loss_fn(y_true, y_pred):
        y_pred = tf.math.l2_normalize(y_pred, -1)
        batch_size = tf.shape(y_pred)[0]    
        
        cut = batch_size // 2 - 1
        x1 = y_pred[:cut:]
        x2 = y_pred[cut::]

        logits = tf.matmul(x1, x2, transpose_b=True) / temperature

        labels = tf.fill((tf.shape(logits)[0],), 0)

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        
        loss = tf.reduce_mean(loss)
        return loss
    return loss_fn
        
def create_model():
    rpa_inp = layers.Input(shape=(None, 1))
    rpa_norm_inp = layers.Normalization()(rpa_inp)
    
    rpa_conv = layers.Conv1D(filters=64, kernel_size=3)(rpa_norm_inp)
    rpa_conv = layers.BatchNormalization()(rpa_conv)
    rpa_conv = layers.Activation("relu")(rpa_conv)
    rpa_conv = layers.MaxPool1D(pool_size=2)(rpa_conv)
    rpa_conv = layers.Conv1D(filters=128, kernel_size=3)(rpa_norm_inp)
    rpa_conv = layers.BatchNormalization()(rpa_conv)
    rpa_conv = layers.Activation("relu")(rpa_conv)
    rpa_conv = layers.MaxPool1D(pool_size=2)(rpa_conv)
    
    rri_inp = layers.Input(shape=(None, 1))
    rri_norm_inp = layers.Normalization()(rri_inp)
    
    rri_conv = layers.Conv1D(filters=64, kernel_size=3)(rri_norm_inp)
    rri_conv = layers.BatchNormalization()(rri_conv)
    rri_conv = layers.Activation("relu")(rri_conv)
    rri_conv = layers.MaxPool1D(pool_size=2)(rri_conv)
    rri_conv = layers.Conv1D(filters=128, kernel_size=3)(rri_norm_inp)
    rri_conv = layers.BatchNormalization()(rri_conv)
    rri_conv = layers.Activation("relu")(rri_conv)
    rri_conv = layers.MaxPool1D(pool_size=2)(rri_conv)
    
    
    merged = layers.Concatenate()([rpa_conv, rri_conv])
    merged_conv = layers.Conv1D(filters=256, kernel_size=3)(merged)
    merged_conv = layers.BatchNormalization()(merged_conv)
    merged_conv = layers.Activation("relu")(merged_conv)
    merged_conv = layers.MaxPool1D(pool_size=2)(merged_conv)
    merged_conv = layers.Conv1D(filters=512, kernel_size=3)(merged_conv)
    merged_conv = layers.BatchNormalization()(merged_conv)
    merged_conv = layers.Activation("relu")(merged_conv)
    merged_conv = layers.MaxPool1D(pool_size=2)(merged_conv)
    
    encoder_out = layers.GlobalAvgPool1D()(merged_conv)
    
    # projection head
    ph = layers.Dense(256)(encoder_out)
    ph = layers.BatchNormalization()(ph)
    ph = layers.Activation("relu")(ph)
    ph_out = layers.Dense(128)(ph)
    
    encoder = Model(inputs=[rpa_inp, rri_inp], outputs=encoder_out)
    model = Model(inputs=[rpa_inp, rri_inp], outputs=ph_out)
    model.compile(
        optimizer = "adam", 
        loss = contrastive_loss(temperature=0.5),
    )

    return encoder, model

encoder, model = create_model() 
show_params(model, "ecg_encoder + projection_head")
weights_path = path.join("res", "ecg_encoder_ah.weights.h5")
model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 1024
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = 100,
    patience = 10,
)
# cb_checkpoint = cbk.ModelCheckpoint(
#     weights_path, 
#     save_best_only = True,
#     save_weights_only = True,
# )
cb_his = HistoryAutosaver(save_path=path.join("history", "ecg_encoder"))
cb_lr = WarmupCosineDecayScheduler(target_lr=0.001, warmup_epochs=5, total_epochs=epochs, min_lr=1e-6)
# cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=15, min_lr=1e-5)
cb_save_encoder = SaveEncoderCallback(encoder, weights_path)

seg_len = 30
extra_seg_len = 10
step_size = 15

ecgs = []
labels = []

p_list = good_p_list()

scaler = StandardScaler()

for idx, p in enumerate(p_list, start=1):
    raw_sig = np.load(path.join("data", f"benhnhan{p}ecg.npy"))
    raw_label = np.load(path.join("data", f"benhnhan{p}label.npy"))[::, :1:].flatten()
    
    sig = clean_ecg(raw_sig)    
    sig = divide_signal(raw_sig, win_size=seg_len*100, step_size=step_size*100)
    label = divide_signal(raw_label, win_size=seg_len, step_size=step_size)
    
    ecgs.append(sig)
    labels.append(label) 
 
ecgs = np.vstack(ecgs)
ecgs = np.array([scaler.fit_transform(e.reshape(-1, 1)).flatten() for e in ecgs])
ecgs = np.vstack([
    ecgs,
    [augment_ecg(e) for e in ecgs]
])

labels = np.vstack(labels)
labels = np.vstack([labels, labels])
labels = np.array([l[extra_seg_len:len(l)-extra_seg_len:] for l in labels])
labels = np.mean(labels, axis=-1)
labels = np.round(labels)

indices = np.arange(len(labels))
train_indices, val_indices = train_test_split(indices, test_size=0.2)

val_ecgs = ecgs[val_indices]
val_labels = labels[val_indices]
ecgs = ecgs[train_indices]
labels = labels[train_indices]

total_samples = len(labels)
print(f"Total samples: {total_samples}\n")
train_generator = data_generator(ecgs, labels, batch_size=batch_size)
val_generator = data_generator(val_ecgs, val_labels, batch_size=batch_size)

steps_per_epoch = total_samples // batch_size
validation_steps = len(val_labels) // batch_size


start_time = timer()
model.fit(
    train_generator,
    epochs = epochs,
    validation_data = val_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_save_encoder],
)
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")