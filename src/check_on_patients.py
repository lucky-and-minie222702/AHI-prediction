from data_functions import *
from model_functions import *
from sklearn.preprocessing import MinMaxScaler

def ECG_structure(name: str):
    # after encoder
    inp = layers.Input(shape=(640, 8)) 
    norm_inp = layers.Normalization()(inp)
    
    conv = ResNetBlock(1, norm_inp, 64, 3, True)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 128, 3, True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 128, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 256, 3, True)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    conv = ResNetBlock(1, conv, 256, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)

    conv = ResNetBlock(1, conv, 512, 3, True)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    conv = ResNetBlock(1, conv, 512, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    conv = ResNetBlock(1, conv, 1024, 3, True)
    conv = ResNetBlock(1, conv, 1024, 3)
    conv = ResNetBlock(1, conv, 1024, 3)
    conv = ResNetBlock(1, conv, 1024, 3)
    
    conv = layers.SpatialDropout1D(rate=0.1)(conv)
    
    se_conv = SEBlock()(conv)

    flat = layers.GlobalAvgPool1D()(se_conv)
    flat = layers.Dense(1024)(flat)
    flat = layers.BatchNormalization()(flat)
    flat = layers.LeakyReLU(negative_slope=0.25)(flat)
    flat = layers.Dropout(rate=0.1)(flat)
    flat = layers.Dense(1024)(flat)
    flat = layers.BatchNormalization()(flat)
    flat = layers.LeakyReLU(negative_slope=0.25)(flat)
    flat = layers.Dropout(rate=0.1)(flat)
    out = layers.Dense(2, activation="softmax")(flat)
    
    model = Model(
        inputs = inp,
        outputs = out,
        name = name
    )
        
    return model

def encoder_structure():
    inp = layers.Input(shape=(600, 10))
    en = layers.Normalization()(inp)
    
    en = layers.Conv1D(filters=32, kernel_size=11, strides=2)(en)
    en = layers.BatchNormalization()(en)
    en = layers.LeakyReLU(negative_slope=0.25)(en)
    en = layers.MaxPool1D(pool_size=3, strides=2)(en)

    en = ResNetBlock(1, en, 64, 9, True)
    en = ResNetBlock(1, en, 64, 9)
    en = ResNetBlock(1, en, 64, 9)
    en = ResNetBlock(1, en, 64, 9)
       
    en = ResNetBlock(1, en, 128, 7, True)
    en = ResNetBlock(1, en, 128, 7)
    en = ResNetBlock(1, en, 128, 7)
    en = ResNetBlock(1, en, 128, 7)
    en = ResNetBlock(1, en, 128, 7)
    
    en = ResNetBlock(1, en, 256, 5, True)
    en = ResNetBlock(1, en, 256, 5)
    en = ResNetBlock(1, en, 256, 5)
    en = ResNetBlock(1, en, 256, 5)
    en = ResNetBlock(1, en, 256, 5)
    
    en = ResNetBlock(1, en, 512, 3, True)
    en = ResNetBlock(1, en, 512, 3)
    en = ResNetBlock(1, en, 512, 3)
    en = ResNetBlock(1, en, 512, 3)
    
    en = ResNetBlock(1, en, 1024, 3, True)
    en = ResNetBlock(1, en, 1024, 3)
    en = ResNetBlock(1, en, 1024, 3)

    en = SEBlock()(en)
    en = layers.Flatten()(en)
    expanded_en = layers.Reshape((640, 8))(en)

    encoder = Model(
        inputs = inp,
        outputs = expanded_en,
    )
    
    return encoder

def count_valid_subarrays(arr, min_length: int, min_separation: int = 0) -> int:
    n = len(arr)
    count = 0
    i = 0
    last_end = -min_separation
    while i <= n - min_length:
        for j in range(i + min_length, n + 1):
            subarray = arr[i:j]
            mean_value = sum(subarray) / len(subarray)

            if round(mean_value) == 1:
                if i >= last_end + min_separation:
                    count += 1
                    last_end = j
                    i = j + min_separation - 1
                    break 
        i += 1 
        print(f"{i}/{n-min_length}", end="r")
    print("\n")
    return count


model_ah = ECG_structure("ah")
model_stage = ECG_structure("stage")
encoder = encoder_structure()

model_id = 1

model_ah.load_weights(path.join("res", f"model_ECG_ah_{model_id}.weights.h5"))
model_stage.load_weights(path.join("res", f"model_ECG_stage_{model_id}.weights.h5"))
encoder.load_weights(path.join("res", "model_auto_encoder_ECG.weights.h5"))

patients_folder = path.join("test_data", "30benhnhannumpy")
res_file = "test_patients_results.txt"
f = open(res_file, "w")

for patient_id in range(1, 29):
    full_ecg = np.load(path.join(patients_folder, f"benhnhan{patient_id}ecg.npy"))
    segmented_ecg = divide_signal(full_ecg, win_size=6000, step_size=100)
    scaler = MinMaxScaler()
    segmented_ecg = scaler.fit_transform(segmented_ecg.T).T  # scale
    segmented_ecg = np.reshape(segmented_ecg, (-1, 600, 10))
    
    segmented_ecg = encoder.predict(segmented_ecg, batch_size=256, verbose=False)
    
    print(f"Analysing paatient {patient_id}...")
    
    # ah
    raw_pred = model_ah.predict(segmented_ecg, batch_size=256, verbose=False)
    ahs = [np.argmax(x) for x in raw_pred]
    # stage
    model_stage.predict(segmented_ecg, batch_size=256, verbose=False)
    wakes = [np.argmax(x) for x in raw_pred]

    ahs_count = count_valid_subarrays(ahs, min_length=10, min_separation=3)
    wakes_count = count_valid_subarrays(wakes, min_length=30, min_separation=0)

    sleep_time = (len(full_ecg) / 100)
    sleep_time -= wakes*30

    ahi = ahs_count / (sleep_time * 60 * 60)

    print(f"Patient {patient_id}:")    
    print(f"Sleep time: {convert_seconds(sleep_time)}")
    print(f"AH count: {ahs_count}")
    print(f"AHI: {ahi}")
    print("="*30)

    print(f"Done patient {patient_id}!")
    
f.close()
