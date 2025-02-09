from data_functions import *
from model_functions import *
import model_framework

# https://www.mdpi.com/2075-4418/14/11/1134

env = model_framework.TrainingEnv(config={
    "max_epoch": 200,
    "num_classes": 2,
    "binary_classification": True,
    "early_stopping_patience": 8,
})
env.summary_env()

def create_model():
    long_inp = layers.Input(shape=)