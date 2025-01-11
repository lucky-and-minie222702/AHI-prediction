from model_functions import *
from data_functions import *

def create_model_ECG_ah():    
    # after encoder
    inp = layers.Input(shape=(1504, 1))  
    norm_inp = layers.Normalization()(inp)
    
    conv = ResNetBlock(1, norm_inp, 64, 3, True)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    
    conv = ResNetBlock(1, conv, 128, 3, True)
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
    
    conv = ResNetBlock(1, conv, 1024, 3, True)
    conv = ResNetBlock(1, conv, 1024, 3)
    conv = ResNetBlock(1, conv, 1024, 3)
    
    se_conv = SEBlock()(conv)
    flat = layers.GlobalAvgPool1D()(se_conv)
    flat = layers.Dense(512)(flat)
    flat = layers.BatchNormalization()(flat)
    flat = layers.LeakyReLU(negative_slope=0.25)(flat)
    out = layers.Dense(1, activation="sigmoid")(flat)
    
    model = Model(
        inputs = inp,
        outputs = out,
    )
        
    return model


def create_model_SpO2_ah():
    inp = layers.Input(shape=(30, 1))
    x = layers.Normalization()(inp)
    
    x = ResNetBlock(1, x, 64, 3, True)
    x = ResNetBlock(1, x, 64, 3)
    x = ResNetBlock(1, x, 64, 3)
    
    x = ResNetBlock(1, x, 128, 3, True)
    x = ResNetBlock(1, x, 128, 3)
    x = ResNetBlock(1, x, 128, 3)
    
    x = ResNetBlock(1, x, 256, 3, True)
    x = ResNetBlock(1, x, 256, 3)
    x = ResNetBlock(1, x, 256, 3) 

    x = ResNetBlock(1, x, 512, 3, True)
    x = ResNetBlock(1, x, 512, 3)
    x = ResNetBlock(1, x, 512, 3) 

    x = SEBlock()(x)
    x = layers.GlobalAvgPool1D()(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(
        inputs = inp,
        outputs = out,
    )
        
    return model

model_ECG = create_model_ECG_ah()
model_SpO2 = create_model_SpO2_ah()

name = sys.argv[sys.argv.index("id")+1]
save_path_ECG = path.join("res", f"model_ECG_ah_{name}.weights.h5")
save_path_SpO2 = path.join("res", f"model_SpO2_ah_{name}.weights.h5")

model_ECG.load_weights(save_path_ECG)
model_SpO2.load_weights(save_path_SpO2)

test_indices = np.load(path.join("patients", "test_indices_ECG_ah.npy"))

sequences_ECG = np.load(path.join("patients", "merged_ECG.npy"))[test_indices]
sequences_SpO2 = np.load(path.join("patients", "merged_SpO2.npy"))[test_indices]
annotations  = np.load(path.join("patients", "merged_anns.npy"))[test_indices]

balance = balancing_data(annotations, 1.0)
combined_balance = np.unique(balance)

sequences_ECG = sequences_ECG[combined_balance]
sequences_Spo2 = sequences_SpO2[combined_balance]
annotations = annotations[combined_balance]

pred_ECG = model_ECG.predict(sequences_ECG, batch_size=128).squeeze()
pred_SpO2 = model_SpO2.predict(sequences_SpO2, batch_size=128).squeeze()

for i in range(1, 10):
    we = round(i / 10, 1)
    ws = round(1 - we, 1)
    
    pred_ECG *= we
    pred_SpO2 *= ws
    pred = pred_ECG + pred_SpO2
    pred = np.round(pred)
    
    accuracy = np.sum(pred == annotations) / len(annotations)
    print(f"we: {we} - ws: {ws} - acc: {accuracy}")
    