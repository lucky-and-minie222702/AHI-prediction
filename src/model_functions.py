import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
import pandas as pd
from keras import Sequential, Model
from keras import layers
from os import path
from keras.saving import load_model 
import argparse
from keras.utils import to_categorical
from keras import optimizers
from sklearn.utils import shuffle
from collections import Counter
from keras import metrics
from sklearn.model_selection import KFold
import sklearn.preprocessing as prep
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sklearn.model_selection as mdselect
import keras.applications as apl
import keras.regularizers as reg
import joblib
import tensorflow.python.keras.backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras.callbacks as cbk
from keras.preprocessing.sequence import pad_sequences
from timeit import default_timer as timer
import random
from typing import *
from tensorflow.keras.utils import Sequence
        
def show_gpus(limit_mem: bool = True):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f" {i:2d} | GPU: {gpu.name}")
    else:
        print("! No GPU detected. Using CPU.")
        
    if limit_mem:
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)  # Only allocate memory as needed
            except RuntimeError as e:
                print(e)

# check for available GPUs
def ResNetBlock(dimension: int, x, filters: int, kernel_size: int, change_sample: bool | int = False, transpose: bool = False, num_layers: int = 2, activation = layers.Activation("relu"), kernel_regularizer=None):
    if not transpose:
        if dimension == 1:
            Conv = layers.Conv1D
        elif dimension == 2:
            Conv = layers.Conv2D
        elif dimension == 3:
            Conv = layers.Conv3D
    else:
        if dimension == 1:
            Conv = layers.Conv1DTranspose
        elif dimension == 2:
            Conv = layers.Conv2DTranspose
        elif dimension == 3:
            Conv = layers.Conv3DTranspose
    
    if isinstance(change_sample, bool):
        strides = 1 + int(change_sample)
    else:
        strides = change_sample
    
    shortcut = x

    x = Conv(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = activation(x)
    
    for _ in range(1, num_layers):
        x = Conv(filters, kernel_size, strides=1, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = activation(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=kernel_regularizer)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = activation(x)
    return x


class SEBlock(layers.Layer):
    def __init__(self, reduction_ratio: int = 4, activation = layers.Activation("relu"), scores_actiation = layers.Activation("sigmoid"), kernel_regularizer=None, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.la = activation  # layers activation
        self.sa = scores_actiation
        self.kr = kernel_regularizer

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.fc1 = layers.Dense(self.channels // self.reduction_ratio, activation=self.la, kernel_regularizer=self.kr)
        self.fc2 = layers.Dense(self.channels, activation=self.sa, kernel_regularizer=self.kr)

    def call(self, inputs):
        input_rank = len(inputs.shape)

        if input_rank == 3:
            se = layers.GlobalAvgPool1D()(inputs)
        elif input_rank == 4:
            se = layers.GlobalAvgPool2D()(inputs)
        elif input_rank == 5: 
            se = layers.GlobalAvgPool3D()(inputs)

        se = self.fc1(se)
        se = self.fc2(se)

        se = tf.reshape(se, [-1] + [1] * (input_rank - 2) + [self.channels])

        return layers.Multiply()([inputs, se])
   

class MyAtt(layers.Layer):
    def __init__(self, depth, num_heads, dropout_rate=0.0):
        super(MyAtt, self).__init__()
        self.num_heads = num_heads
        self.depth = depth
        self.d_model = depth * num_heads
        
        # Query, Key, Value projections
        self.Wq = layers.Dense(self.d_model)
        self.Wk = layers.Dense(self.d_model)
        self.Wv = layers.Dense(self.d_model)
        
        # Output projection
        self.dense = layers.Dense(self.d_model)
        
        # Normalization and dropout
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization()
    
    def split_heads(self, x, batch_size):
        """Split into multiple heads and reshape."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, num_heads, seq_len, depth)
    
    def scaled_dot_product_attention(self, Q, K, V):
        """Compute attention scores."""
        matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (batch, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)  # (batch, num_heads, seq_len_q, depth)
        return output, attention_weights

    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]

        # Linear projections
        Q = self.split_heads(self.Wq(query), batch_size)
        K = self.split_heads(self.Wk(key), batch_size)
        V = self.split_heads(self.Wv(value), batch_size)

        # Scaled dot-product attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.num_heads * self.depth))

        # Final dense projection
        output = self.dense(concat_attention)
        output = self.dropout(output)
        return self.norm(output)


class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs = {}):
        self.logs=[]

    def on_epoch_begin(self, epoch: int, logs = {}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch: int, logs = {}):
        self.logs.append(timer()-self.starttime)


# Attention mechanism  
class MyMultiHeadRelativeAttention(layers.Layer):
    def __init__(self, depth: int, num_heads: int, max_relative_position: int, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.d_model = depth * num_heads
        self.max_relative_position = max_relative_position
        
        self.relative_embedding = layers.Embedding(2 * max_relative_position + 1, self.depth)
        
        self.query_dense = layers.Dense(self.d_model)
        self.key_dense = layers.Dense(self.d_model)
        self.value_dense = layers.Dense(self.d_model)
        
        self.output_dense = layers.Dense(self.d_model)
        self.softmax = layers.Softmax(axis=-1)

    def split_heads(self, x, batch_size):
        return tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

    def call(self, inputs):
        x = inputs  # batch_size, seq, d_model
        batch_size = tf.shape(x)[0]
        seq = tf.shape(x)[1]

        # Generate relative positions
        range_vec = tf.range(seq)
        relative_positions = tf.clip_by_value(
            range_vec[:, None] - range_vec[None, :], 
            -self.max_relative_position, self.max_relative_position
        )
        relative_indices = relative_positions + self.max_relative_position  

        relative_embeddings = self.relative_embedding(relative_indices)  # seq, seq, depth

        Q = self.query_dense(x)  # batch_size, seq, d_model
        K = self.key_dense(x)
        V = self.value_dense(x)
  
        Q = self.split_heads(Q, batch_size)  # batch, seq, num_heads, depth
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # batch_size, num_heads, seq, depth
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        scaling_factor = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        content_scores = tf.matmul(Q, K, transpose_b=True) / scaling_factor  # batch, num_heads, seq, seq

        rel_scores = tf.einsum('bhqd,qkd->bhqk', Q, relative_embeddings) / scaling_factor  # batch, num_heads, seq, seq  

        combined_scores = content_scores + rel_scores

        attention_weights = self.softmax(combined_scores)  # batch, num_heads, seq, seq

        attention_output = tf.matmul(attention_weights, V)  # batch, num_heads, seq, depth

        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])  # batch, num_heads, seq, depth
        attention_output = tf.reshape(attention_output, (batch_size, seq, self.d_model))

        output = self.output_dense(attention_output)
        return output

  
class SaveEncoderCallback(tf.keras.callbacks.Callback):
    def __init__(self, encoder, save_path: str, save_after_epoch: int = 1):
        super(SaveEncoderCallback, self).__init__()
        self.encoder = encoder
        self.save_path = save_path
        self.sae = save_after_epoch

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.sae == 0:
            self.encoder.save_weights(self.save_path)

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs = {}):
        self.logs=[]

    def on_epoch_begin(self, epoch: int, logs = {}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch: int, logs = {}):
        self.logs.append(timer()-self.starttime)
        
class HistoryAutosaver(keras.callbacks.Callback):
    def __init__(self, save_path: str):
        self.p = save_path
        self.history = {}

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            np.save(self.p + f"_{key}", np.array(self.history[key]))
            
def convert_bytes(byte_size: int) -> str:
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = byte_size
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def convert_seconds(total_seconds: float) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def show_params(model, name: str = "Keras_deep_learning"):
    print(f"Model {name}:")
    params = model.count_params()
    print(" | Total params :", "{:,}".format(params).replace(",", " "))
    print(" | Size         :", convert_bytes(params * 4))
    
def show_data_size(train: np.ndarray, test: np.ndarray, val: np.ndarray):
    data = [train, test, val]
    labels = ["Train", "Test", "Validation"]
    for i in range(3):
        cls, counts = np.unique(data[i], return_counts=True)
        print(f"{labels[i]} set:")
        for idx in range(len(cls)):
            print(f" | Class {cls[idx]}: {counts[idx]}")
    

class WarmupCosineDecayScheduler(cbk.Callback):
    def __init__(self, warmup_epochs, total_epochs, target_lr, min_lr):
        super(WarmupCosineDecayScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = (epoch + 1) / self.warmup_epochs * self.target_lr
        else:
            # Cosine Decay
            decay_epoch = epoch - self.warmup_epochs
            decay_total = self.total_epochs - self.warmup_epochs
            lr = self.min_lr + (self.target_lr - self.min_lr) * (1 + np.cos(np.pi * decay_epoch / decay_total)) / 2
            
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        
## 
def create_ecg_encoder():
    inp = layers.Input(shape=(1000, 1))
    norm_inp = layers.Normalization()(inp)
    
    ds_conv = layers.Conv1D(filters=64, kernel_size=7, strides=2, padding="same")(norm_inp)
    ds_conv = layers.BatchNormalization()(ds_conv)
    ds_conv = layers.Activation("relu")(ds_conv)
    ds_conv = layers.MaxPool1D(pool_size=2)(ds_conv)
    
    conv = ResNetBlock(1, ds_conv, 64, 3)
    conv = ResNetBlock(1, conv, 64, 3)
    conv = ResNetBlock(1, conv, 128, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 128, 3)
    conv = ResNetBlock(1, conv, 256, 3, change_sample=True)
    conv = ResNetBlock(1, conv, 256, 3)
    
    fc = layers.GlobalAvgPool1D()(conv)
    out = layers.Dense(256)(fc)
    
    model = Model(inputs=inp, outputs=out)
    model.load_weights(path.join("res", "ecg_encoder.weights.h5"))
    
    return model
    
    

def prototypical_loss(support_set, query_sample):
    support_means = tf.math.reduce_mean(support_set, axis=1)  # Compute class prototypes
    dists = tf.norm(query_sample - support_means, axis=1)  # Compute distances
    return tf.nn.softmax(-dists)  # Class probabilities

def generate_support_query_sets(X, y, num_classes, num_samples_per_class):
    support_set, support_labels = [], []
    
    for cls in range(num_classes):
        indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
        support_set.append(X[sampled_indices])
        support_labels.append(np.full((num_samples_per_class,), cls))
    
    return np.array(support_set), np.array(support_labels)

def predict_using_ecg_encoder(ecg_encoder, X_ecg, y_labels, X_new, num_sample_per_class):
    support_ecgs, _ = generate_support_query_sets(X_ecg, y_labels, num_classes=2, num_samples_per_class=num_sample_per_class)
    # Convert to TensorFlow format
    support_ecgs = tf.convert_to_tensor(support_ecgs)
    query_ecg = tf.convert_to_tensor([X_new])
    cls0 = ecg_encoder(support_ecgs[0])
    print(support_ecgs[0].shape)
    cls1 = ecg_encoder(support_ecgs[1])
    probs = prototypical_loss(tf.concat([cls0, cls1], axis=0), ecg_encoder(query_ecg))
    return probs.numpy()