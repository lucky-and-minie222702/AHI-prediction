import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.saving import load_model
from keras.utils import plot_model
from os import path

ECG_path = path.join("res", "model_ECG.keras")
ECG_model = load_model(ECG_path)
sECG_path = path.join("summary", "model.png")
plot_model(ECG_model, sECG_path, dpi=500, show_layer_activations=True, show_shapes=True)
