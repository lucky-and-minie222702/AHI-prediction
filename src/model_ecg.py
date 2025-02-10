from data_functions import *
from model_functions import *
import model_framework

env = model_framework.TrainingEnv(config={
    "name": "ecg",
    "max_epoch": 50,
    "regression": True,
    "early_stopping_patience": 5,
    "early_stopping_epoch": 20,
    
    "weights_dir": "weights",
    "logs_dir": "history",
})

num_p = 33
info = open(path.join("data", "info.txt"), "r").readlines()
p_list = []
no_spo2 = []

for s in info:
    s = s[:-1:]
    if "*" in s:
        p_list.append(int(s[1::]))
        no_spo2.append(int(s[1::]))
    else:
        p_list.append(int(s))
        
        
def create_model():
    inp_rpa = layers.Input(shape=(None, 1))
    norm_inp_rpa = layers.Normalization()(inp_rpa)
    conv_rpa = layers.Conv1D(filters=32, kernel_size=3, padding="same")(norm_inp_rpa)
    conv_rpa = layers.BatchNormalization()(conv_rpa)
    conv_rpa = layers.Activation("relu")(conv_rpa)
    conv_rpa = layers.MaxPool1D(pool_size=3, strides=2)(conv_rpa)
    conv_rpa = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv_rpa)
    conv_rpa = layers.BatchNormalization()(conv_rpa)
    conv_rpa = layers.Activation("relu")(conv_rpa)
    conv_rpa = layers.MaxPool1D(pool_size=3, strides=2)(conv_rpa)
    
    inp_rri = layers.Input(shape=(None, 1))
    norm_inp_rri = layers.Normalization()(inp_rri)
    conv_rri = layers.Conv1D(filters=32, kernel_size=3, padding="same")(norm_inp_rri)
    conv_rri = layers.BatchNormalization()(conv_rri)
    conv_rri = layers.Activation("relu")(conv_rri)
    conv_rri = layers.MaxPool1D(pool_size=3, strides=2)(conv_rri)
    conv_rri = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv_rri)
    conv_rri = layers.BatchNormalization()(conv_rri)
    conv_rri = layers.Activation("relu")(conv_rri)
    conv_rri = layers.MaxPool1D(pool_size=3, strides=2)(conv_rri)
    
    r_peak_features = layers.Concatenate(axis=-2)([conv_rpa, conv_rri])
    r_peak_features = layers.Conv1D(filters=128, kernel_size=3)(r_peak_features)
    r_peak_features = layers.BatchNormalization()(r_peak_features)
    r_peak_features = layers.Activation("relu")(r_peak_features)
    r_peak_features = layers.MaxPool1D(pool_size=3, strides=2)(r_peak_features)
    r_peak_features = layers.Conv1D(filters=256, kernel_size=3)(r_peak_features)
    r_peak_features = layers.BatchNormalization()(r_peak_features)
    r_peak_features = layers.Activation("relu")(r_peak_features)
    r_peak_features = layers.MaxPool1D(pool_size=3, strides=2)(r_peak_features)
    r_peak_features = SEBlock()(r_peak_features)
    
    inp = layers.Input(shape=(None, 1))
    norm_inp = layers.Normalization()(inp)
    
    conv = layers.Conv1D(filters=64, kernel_size=11, strides=2,padding="same")(norm_inp)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.MaxPool1D(pool_size=3, strides=2)(conv)
    
    conv = ResNetBlock(1, conv, 64, 9)
    conv = ResNetBlock(1, conv, 64, 7)
    conv = ResNetBlock(1, conv, 64, 5)
    
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    
    conv = layers.Concatenate(axis=-2)([conv, r_peak_features])
    
    conv = ResNetBlock(1, conv, 512, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    
    rnn = layers.ConvLSTM1D(filters=512, kernel_size=3, strides=2, return_sequences=True)(conv)
    rnn = layers.ConvLSTM1D(filters=512, kernel_size=3, strides=2, return_sequences=True)(rnn)
    
    se = SEBlock()(rnn)
    
    fc = layers.GlobalAvgPool1D()(se)
    fc = layers.Dense(512)(fc)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Activation("relu")(fc)
    out = layers.Dense(1)(fc)    
    
    
    model = Model(inputs=[inp, inp_rpa, inp_rri], outputs=out)
    model.compile(
        optimizer = "adam",
        loss = "mse",
        metrics = ["mae"]
    )

    return model

model = create_model()