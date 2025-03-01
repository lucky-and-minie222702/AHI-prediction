from data_functions import *
from model_functions import *
# import model_framework
from sklearn.preprocessing import MinMaxScaler, StandardScaler

show_gpus()

def augment_ecg(signal):
    signal = time_warp(signal, sigma=0.075)
    signal = add_noise(signal, noise_std=0.05)
    signal = time_shift(signal, shift_max=20)
    signal *= np.random.randint(80, 120) / 100
    return signal

def data_generator(X, y, X_aug, batch_size):
    def generator():
        while True:
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_indices = indices[start:end]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                X_aug_batch = X_aug[batch_indices]
                y_aug_batch = y_batch
                yield np.concatenate([X_batch, X_aug_batch], axis=0), np.concatenate([y_batch, y_aug_batch], axis=0)
    
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, *X.shape[1:]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *y.shape[1:]), dtype=tf.float32)
    ))


def contrastive_loss_with_augment(temperature):
    def loss_fn(y_true, y_pred):
        hidden = y_pred
        
        hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]
        
        labels = tf.range(batch_size)

        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature

        loss_aa = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_aa, from_logits=True)
        loss_ab = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_ab, from_logits=True)
        loss_bb = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_bb, from_logits=True)
        loss_ba = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_ba, from_logits=True)
        
        
        loss = tf.reduce_mean(loss_aa + loss_ab + loss_bb + loss_ba)
        return loss
    return loss_fn


def contrastive_loss_no_augment(temperature):
    def loss_fn(y_true, y_pred):
        hidden = y_pred
        
        hidden = tf.math.l2_normalize(hidden, -1)

        logits= tf.matmul(hidden, hidden, transpose_b=True) / temperature
        
        labels = tf.range(batch_size)
        
        raw_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss = tf.reduce_mean(raw_loss)
        
        return loss
    return loss_fn

        
def create_model():
    inp = layers.Input(shape=(3000, 1))
    norm_inp = layers.Normalization()(inp)
    
    ds_conv = layers.Conv1D(filters=64, kernel_size=7, strides=2, padding="same")(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=2)(ds_conv)
    
    conv = ResNetBlock(1, ds_conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 1024, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 1024, 3)
    conv = ResNetBlock(1, conv, 1024, 3)

    encoder_out = layers.GlobalAvgPool1D()(conv)
    
    # projection head
    ph = layers.Dense(128)(encoder_out)
    ph = layers.BatchNormalization()(ph)
    ph = layers.Activation("relu")(ph)
    ph_out = layers.Dense(32)(ph)
    
    encoder = Model(inputs=inp, outputs=encoder_out)
    model = Model(inputs=inp, outputs=ph_out)
    model.compile(
        optimizer = "adam", 
        loss = contrastive_loss_no_augment(temperature=0.5),
    )

    return encoder, model

encoder, model = create_model() 
show_params(model, "ecg_encoder + projection_head")
weights_path = path.join("res", "ecg_encoder.weights.h5")
model.save_weights(weights_path)

epochs = 200 if not "epochs" in sys.argv else int(sys.argv[sys.argv.index("epochs")+1])

batch_size = 512
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
# cb_lr = cbk.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-5)
cb_save_encoder = SaveEncoderCallback(encoder, weights_path)

seg_len = 30
extra_seg_len = 0
step_size = 30

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

labels = np.vstack(labels)
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
# train_generator = data_generator(ecgs, labels, np.array([augment_ecg(e) for e in ecgs]), batch_size=batch_size)
# val_generator = data_generator(val_ecgs, val_labels, np.array([augment_ecg(e) for e in val_ecgs]), batch_size=batch_size)

# steps_per_epoch = total_samples // batch_size
# validation_steps = len(val_labels) // batch_size


start_time = timer()
model.fit(
    ecgs,
    labels,
    epochs = epochs,
    validation_data = (val_ecgs, val_labels),
    # steps_per_epoch = steps_per_epoch,
    # validation_steps = validation_steps,
    callbacks = [cb_early_stopping, cb_his, cb_lr, cb_save_encoder],
)
total_time = timer() - start_time
print(f"Training time {convert_seconds(total_time)}")