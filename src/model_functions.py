import numpy as np
import keras
import sys
import os
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
        
def show_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"!!! GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f" {i:2d} | GPU: {gpu.name}")
    else:
        print("! No GPU detected. Using CPU.")

# check for available GPUs
def ResNetBlock(dimension: int, x, filters: int, kernel_size: int, change_sample: bool = False, transpose: bool = False, activation = layers.Activation("relu")):
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

    x = Conv(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation(x)

    x = Conv(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv(filters, kernel_size, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = activation(x)
    return x


class SEBlock(layers.Layer):
    def __init__(self, reduction_ratio: int = 2, activation = layers.Activation("relu"), scores_actiation = layers.Activation("sigmoid"), **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.la = activation  # layers activation
        self.sa = scores_actiation

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.fc1 = layers.Dense(self.channels // self.reduction_ratio, activation=self.la)
        self.fc2 = layers.Dense(self.channels, activation=self.sa)

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
    def __init__(self, depth: int, num_heads: int, seq_len: int = -1):
        super(MyAtt, self).__init__()
        self.num_heads = num_heads        
        self.depth = depth
        self.d_model = depth * num_heads
        self.seq_len = seq_len
        
        self.Wq = tf.keras.layers.Dense(self.d_model)
        self.Wk = tf.keras.layers.Dense(self.d_model)
        self.Wv = tf.keras.layers.Dense(self.d_model)
        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size: int):
        # num_heads, depth
        x = tf.reshape(x, (batch_size, self.seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # batch, heads, seq, depth

    def scaled_dot_product_attention(self, Q, K, V):
        matmul_qk = tf.matmul(Q, K, transpose_b=True) 
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # softmax normalized
        output = tf.matmul(weights, V)
        return output, weights

    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]
        
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)
        
        # split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        attention_output, w = self.scaled_dot_product_attention(Q, K, V)
        
        # merge heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, self.seq_len, self.d_model))
        
        output = self.dense(concat_attention)
        return output


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
