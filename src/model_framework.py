"""
Simple framework for better training and command-line interacting experience with keras model
"""

import os
from typing import *
import argparse
import math
from os import path
import numpy as np
import keras
import keras.callbacks as cbk
from timeit import default_timer as timer
import joblib
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error, root_mean_squared_error
from keras.utils import to_categorical
import tensorflow as tf

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs = {}):
        self.logs=[]

    def on_epoch_begin(self, epoch: int, logs = {}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch: int, logs = {}):
        self.logs.append(timer()-self.starttime)
        
class HistoryAutosaver(keras.callbacks.Callback):
    def __init__(self, save_dir: str, model_name: str = "", session_id: int = None):
        self.dir = save_dir
        self.model_name = model_name
        self.session_id = ""
        if session_id is not None:
            self.session_id = str(session_id)
        self.history = {}

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            np.save(path.join(self.dir, f"{self.model_name}_{self.session_id}_{key}"), np.array(value))
            
 
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
    
def equally_dis(num_classes: int) -> List[int]:
    ele = 1.0 / num_classes
    last_ele = 1 - ele * (num_classes - 1)
    return [ele for _ in range(num_classes - 1)] + [last_ele]
    
def split_classes(y: np.ndarray, class_ratio: List[float] = None, max_total_samples: int = None):
    class_counts = np.unique(y, return_counts=True)[1]
    num_classes = len(class_counts)
    if class_ratio is None:
        class_ratio = equally_dis(num_classes)
    
    # bin search
    l = 0
    r = len(y)
    m = 0
    while l <= r:
        m = (l+r)//2
        wrong = any([(m * class_ratio[i]) > class_counts[i] for i in range(num_classes)])
        if wrong:
            r = m -1
        else:
            l = m + 1
    
    total_samp = m
    if max_total_samples is not None:
        total_samp = min(total_samp, max_total_samples)

    class_idx = []
    for cls, i in enumerate(class_ratio):
        k = int(total_samp * i)
        class_idx.extend(np.random.choice(np.where(y==cls)[0], k, replace=False))
        
    return np.array(class_idx), np.array([int(total_samp * r) for r in class_ratio])

def print_confusion_matrix(cm: np.ndarray | List[List[int]], labels = None):
    if labels is None:
        labels = list(map(str, range(len(cm))))
    assert max([len(l) for l in labels]) <= 6, "Labels length for confusion matrix must not exceed 6"
    print("Confusion Matrix:")
    print(" " * 10, "Predicted", sep="")
    print(" " * 10, " ".join(f"{label:>6}" for label in labels), sep="")
    print(" " * 10 + "-" * (7 * (len(labels))), sep="")
    remain_label = "Actual"
    for i, row in enumerate(cm):
        print(f"{remain_label[i] if i < len(remain_label) else " "} {labels[i]:>6} |", " ".join(f"{val:>6}" for val in row), " |", sep="")
    print(remain_label[len(cm)] if len(cm) < len(remain_label) else " ", " " * 9, "-" * (7 * (len(labels))), sep="")
    for s in remain_label[len(cm)+1::]:
        print(s)

class Tee:
    def reset():
        sys.stdout.file.close() 
        sys.stdout = sys.__stdout__
    
    # tee-like behaviour
    def __init__(self, file: str):
        self.file = file
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class TrainingEnv:
    """
    Training environment for keras
    Arguments:
        - config: Environment configuration. See TrainingEnv.DEFAULT_CONFIG for default configuration
        - using_command_line: active command-line mode which allow modified current configuration
        - strict_rario: strictly check every ratio option

    Configuration includes:
        - model_name: name for the model (default = Keras_deep_learning)
        - batch_size: batch size
        - max_epoch: maximum epoch for training
        - early_stopping_epoch: epoch which the early stoppping callback will monitor
        - early_stopping_patience: early stopping epoch patience
        - early_stopping_monitor: monitor for early stopping
        - early_stopping_mode: mode for early stopping
        - verbose: verbose mode for keras model training API
        - regression: enable regression model mode
        - num_classes: number of classes
        - class_ratio: class ratio (total samples may be different with each ratio configuration)
        - binary_classification: enable binary classification mode (does not same as 2-class one-hot-encoded)
        - dataset_dir: dataset directory path
        - input_files: dataset files for input
        - output_files: dataset files for labels
        - data_ratio: ratio for [train, val, test] data
        - weights_dir: directory to save best weights for every session
        - logs_dir: directory to save logs for every session
        - no_logs: do not show any logs from tensorflow
        - disable_GPU: disable GPU, only using CPU
        - disable_XLA: disable XLA (Accelerated Linear Algebra)
        - lazy_loading: enable lazy_loading
        - train: active training mode (only required if require_activation=True)
    """
    
    DEFAULT_CONFIG = {
        "model_name": "Keras_deep_learning",
        "batch_size": 128,
        "max_epoch": None,
        "early_stopping_epoch": None,
        "early_stopping_patience": 5,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "verbose": 1,
        "regression": False,
        "num_classes": None,
        "class_ratio": None,
        "binary_classification": False,
        "dataset_dir": "",
        "input_files": None,
        "output_files": None,
        "data_ratio": [0.7, 0.2, 0.1],
        "weights_dir": "",
        "logs_dir": "",
        "callbacks": [],
        "require_activation": False,
        # tensorflow options
        "no_logs": False,
        "disable_GPU": False,
        "disable_XLA": False,
        "lazy_loading": False,
    }
    
    @staticmethod
    def default_classification_test_function(y_true, y_pred, labels = None, binary_classification: bool = False, num_thresholds: int = 9):
        thresholds = np.linspace(0, 1, num_thresholds+2)[1:-1:]
        def show_res(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            print_confusion_matrix(cm, labels=labels)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
            if cm.shape[-1] == 2:  # 2-class
                tn, fp, fn, tp = cm.ravel()
                # Compute classification metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred,zero_division=0)  # Sensitivity
                specificity = tn / (tn + fp)  # True Negative Rate
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)  # Matthews Correlation Coefficient
                print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
                # Print evaluation metrics
                print("Evaluation Metrics:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall (Sensitivity): {recall:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"ROC AUC: {auc:.4f}")
                print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        if binary_classification:
            for threshold in thresholds:
                threshold = round(threshold, 2)
                print(f"THRESHOLD {threshold}:")
                show_res(y_true, np.where(y_pred >= threshold, 1, 0))
                print("\n")
        else:
            show_res(y_true, y_pred)
            
    @staticmethod
    def default_regression_test_function(y_true, y_pred):
        # Number of data points and predictors (for Adjusted R²)
        n = len(y_true)
        p = 1  # Modify this if using multiple features in a regression model

        # Compute regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Adjusted R²
        mape = mean_absolute_percentage_error(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)

        # Print evaluation metrics
        print("Regression Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Adjusted R² Score: {adj_r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
        print(f"Median Absolute Error: {median_ae:.4f}")

        
    def __init__(self, config: Dict[str, Any] = {}, using_command_line: bool = True, strict_ratio: bool = True):
        assert not any([key not in self.DEFAULT_CONFIG for key in config.keys()]), "Every key in config must match the DEFAULT_CONFIG"
        self.__config = self.DEFAULT_CONFIG
        self.__config.update(config)
        self.shortcut = {}

        parser = argparse.ArgumentParser(
            prog='NHCT',
            description='Keras training framework',
            epilog="Made for easier training")
        
        parser.add_argument("-n", "--model_name", help="name for the model (default = Keras_deep_learning)", type=str)
        parser.add_argument("-bs", "--batch_size", help="batch size", type=int)
        parser.add_argument("-me", "--max_epoch", help="maximum epoch for training", type=int)
        parser.add_argument("-ese", "--early_stopping_epoch", help="epoch which the early stoppping callback will monitor", type=int)
        parser.add_argument("-esp", "--early_stopping_patience", help="early stopping epoch patience", type=int)
        parser.add_argument("-esmn", "--early_stopping_monitor", help="monitor for early stopping", type=str)
        parser.add_argument("-esmd", "--early_stopping_mode", help="mode for early stopping", type=str)
        parser.add_argument("-vb", "--verbose", help="verbose mode for keras model training API", type=int)
        parser.add_argument("-r", "--regression", help="enable regression model mode", action="store_true")
        parser.add_argument("-nc", "--num_classes", help="number of classes", type=int)
        parser.add_argument("-cr", "--class_ratio", help="class ratio (total samples may be different with each ratio configuration)", type=str)
        parser.add_argument("-bc", "--binary_classification", help="enable binary classification mode (does not same as 2-class one-hot-encoded)", action="store_true")
        parser.add_argument("-dd", "--dataset_dir", help="dataset directory path", type=str)
        parser.add_argument("-di", "--input_files", help="dataset files for input", type=str)
        parser.add_argument("-do", "--output_files", help="dataset files for labels", type=str)
        parser.add_argument("-dr", "--data_ratio", help="ratio for [train, val, test] data", type=str)
        parser.add_argument("-wd", "--weights_dir", help="directory to save best weights for every session", type=str)
        parser.add_argument("-ld", "--logs_dir", help="directory to save logs for every session", type=str)
        # parser.add_argument("-id", "--session_id", help="id for this training session", type=str)
        parser.add_argument("-nl", "--no_logs", help="do not show any logs from tensorflow", action="store_true")
        parser.add_argument("-dg", "--disable_GPU", help="disable GPU, only using CPU", action="store_true")
        parser.add_argument("-dx", "--disable_XLA", help="disable XLA (Accelerated Linear Algebra)", action="store_true")
        parser.add_argument("-ll", "--lazy_loading", help="enable lazy_loading", action="store_true")
        parser.add_argument("-t", "--train", help="active training mode (only required if require_activation=True)", action="store_true")
            
        # only use to generate docstring
        # for action in parser._actions:
        #     print(f" - {action.dest}: {action.help}")
        # exit()
            
        args = parser.parse_known_args()[0]
        args = vars(args)
        self.active_train = args["train"]
        args.pop("train")

        for key, value in args.items():
            if isinstance(value, bool):
                if args[key] == value:  # still in Default
                    args[key] = None

        args = {k: v for k, v in args.items() if v is not None}
        if using_command_line:
            self.__config.update(args)
            if self.__config["no_logs"]:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            if self.__config["disable_GPU"]:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            if self.__config["disable_XLA"]:
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
            if self.__config["lazy_loading"]:
                os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        assert self.__config["max_epoch"] is not None, "Maximum epoch must be specified"
        
        if self.__config["early_stopping_epoch"] is None:
            self.__config["early_stopping_epoch"] = self.__config["max_epoch"] // 2
        
        if not self.__config["regression"]:
            assert self.__config["num_classes"] is not None, "Number of classes must be specified"
            if self.__config["class_ratio"] is None:
                self.__config["class_ratio"] = equally_dis(self.__config["num_classes"])
            else:
                if isinstance(self.__config["class_ratio"], str):
                    self.__config["class_ratio"] = list(map(float, self.__config["class_ratio"].split(",")))
            if strict_ratio:
                assert math.isclose(sum(self.__config["class_ratio"]), 1.0), "Sum of class ratio must be 1.0"
            assert len(self.__config["class_ratio"]) == self.__config["num_classes"], "Class ratio must match with the number of classes"
            
        if self.__config["binary_classification"]:
            assert self.__config["num_classes"] == 2, f"Binary classification mode only available if number of classes is 2"
        
        if isinstance(self.__config["data_ratio"], str):
            self.__config["data_ratio"] = list(map(float, self.__config["data_ratio"].split(",")))
        assert len(self.__config["data_ratio"]) == 3, "Data ratio must match [train, test, validation]"
        if strict_ratio:
            assert math.isclose(sum(self.__config["data_ratio"]), 1.0), "Sum of data ratio muse be 1.0"
        
        if isinstance(self.__config["input_files"], str):
            self.__config["input_files"] = self.__config["input_files"].split(",")
        if isinstance(self.__config["output_files"], str):
            self.__config["output_files"] = self.__config["output_files"].split(",")

        
    @property
    def config(self):
        return self.__config
    
    def summary_env(self):
        idx = 0
        print(f"\n   Option                    Value")
        for key, value in self.__config.items():
            if key == "callbacks":
                continue
            idx += 1
            print(f" - {key:<24}: {value}")
        print()
    
    def update_config(self, new_config: Dict[str, any]):
        new_val = {}
        for key, value in new_config.items():
            if key in self.__config:
                new_val[key] = value
        self.__config.update(new_val)
    
    def deploy(self, model: keras.Model | keras.Sequential, inputs, outputs, session_id: str, test: bool = True, save_logs: bool = True, auto_categorize: bool = True, auto_check_num_classes: bool= True, compile_options = None, main_output: int = 0, test_labels = None, test_function_options = {}, require_activation: bool = True) -> str:
        """
        Deploy a keras model into the environment with current configuration and given input, output data
        
        Arguments:
            - model: keras model
            - inputs: input data (does not need of using input_files)
            - outputs: input data (does not need of using output_files)
            - session_id: id for this training session
            - test: running test after training. Defaults to True
            - save_logs: saving logs while running training. Defaults to True
            - auto_categorize: Automatically convert main output to one-hot-encoded data. Defaults to True
            - auto_check_num_classes (bool, optional): Automatically check whether the num_classes match the actual number of classes of the main output. Defaults to True
            - compile_options: Compile option to compile model, None means no need to compile. Defaults to None
            - main_output (int, optional): Main output index (no need to specific if you do not have multipel output model). Defaults to 0
            - test_labels (_type_, optional): Test labels for logs and report (maximum length for each label is 6). Defaults to None
            - test_function_options: Options for the test function. Default to {}. Includes:
                - num_thresholds: number of thresholds to report, each threshold will be equally divided between 0 and 1 and (always avoid 0 and 1 threshold)
            - require_activation: Require -t or --train flag to active training mode (to run the training code). Defaults to True
            
        Return:
            - Model best weights path
        """
        if require_activation:
            if not self.active_train:
                print("You have not active training mode yet! (using -t or --train, using -h or --help to see help)")
                return

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f" {i+1:2d} | GPU: {gpu.name}")
        else:
            print("! No GPU detected. Using CPU.")
        print()

        if compile_options is not None:
            model.compile(**compile_options)
        
        if self.__config["input_files"] is not None:
            for fn in self.__config["input_files"]:
                p = path.join(self.__config["dataset_dir"], fn) 
                if not p.endswith(".npy"):
                    p + ".npy"
                inputs.append(np.load(p))
        if self.__config["output_files"] is not None:
            for fn in self.__config["output_files"]:
                p = path.join(self.__config["dataset_dir"], fn)
                if not p.endswith(".npy"):
                    p + ".npy"
                outputs.append(np.load(p))

        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        if auto_check_num_classes:
            # num_classes == classes of the main output
            actual_num_classes = len(np.unique(outputs[main_output])) if len(outputs[main_output].shape) == 1 else outputs[main_output].shape[-1]
            assert self.__config["num_classes"] == actual_num_classes , f"Number of classes in config which is {self.__config["num_classes"]} must match the actual number of classes in the dataset"
            
        if auto_categorize and not self.__config["binary_classification"]:
            if len(outputs[main_output].shape) == 1:
                outputs[main_output] = to_categorical(outputs[main_output], num_classes=self.__config["num_classes"])
            
        argmax = lambda x: np.argmax(x, axis=-1) if not self.__config["binary_classification"] else x
            
        total_samples = len(outputs[main_output])
            
        cb_early_stopping = cbk.EarlyStopping(
            restore_best_weights = True,
            start_from_epoch = self.__config["early_stopping_epoch"],
            patience = self.__config["early_stopping_patience"],
        )
        weights_path = path.join(self.__config["weights_dir"], f"{self.__config["model_name"]}_{session_id}.weights.h5")
        cb_checkpoint = cbk.ModelCheckpoint(
            weights_path, 
            save_best_only = True,
            save_weights_only = True,
            monitor = self.__config["early_stopping_monitor"],
            mode = self.__config["early_stopping_mode"],
        )
        cb_timer = TimingCallback()
        cb_history_saver = HistoryAutosaver(
            save_dir = self.__config["logs_dir"],
            model_name = self.__config["model_name"], 
            session_id = session_id
        )
        callbacks = [cb_early_stopping, cb_checkpoint, cb_timer] + self.__config["callbacks"]
        if save_logs:
            callbacks.append(cb_history_saver)
        
        indices = np.arange(total_samples)
        if not self.__config["regression"]:
            indices = split_classes(outputs[main_output])[0]
        indices = np.random.permutation(indices)
        train_size = int(total_samples * self.__config["data_ratio"][0])
        test_size = int(total_samples * self.__config["data_ratio"][1])
        val_size = int(total_samples * self.__config["data_ratio"][1])
        train_indices = indices[:train_size:]
        test_indices = indices[train_size:train_size+test_size:]
        val_indices = indices[train_size+test_size:]
        
        if save_logs:
            np.save(path.join(self.__config["logs_dir"], f"train_indices_{self.__config["model_name"]}_{session_id}"), train_indices)
            np.save(path.join(self.__config["logs_dir"], f"test_indices_{self.__config["model_name"]}_{session_id}"), test_indices)
            np.save(path.join(self.__config["logs_dir"], f"val_indices_{self.__config["model_name"]}_{session_id}"), val_indices)

        multiple_input = len(inputs) > 1
        multiple_output = len(outputs) > 1

        X_train, y_train = [], []
        X_test, y_test = [], []
        X_val, y_val = [], []
        for data in inputs:
            X_train.append(data[train_indices])
            X_test.append(data[test_indices])
            X_val.append(data[val_indices])
        for data in outputs:
            y_train.append(data[train_indices])
            y_test.append(data[test_indices])
            y_val.append(data[val_indices])
        
        if not self.__config["regression"]:
            unique_classes, class_counts = np.unique(argmax(y_train[main_output]), return_counts=True)
            class_weights = {cls: 1.0 / count for cls, count in zip(unique_classes, class_counts)}
            sample_weight = np.array([class_weights[y] for y in argmax(y_train[main_output])])
        
        log_file = open(path.join(self.__config["logs_dir"], f"log_{self.__config["model_name"]}_{session_id}.txt"), "w")
        
        sys.stdout = Tee(log_file)
        print(f"Model {self.__config["model_name"]}_{session_id}")
        print()
        show_params(model, name=self.__config["model_name"])
        print()
        show_data_size(y_train[main_output], y_test[main_output], y_val[main_output])
        print()
        Tee.reset()
        
        model.fit(
            x = X_train if multiple_input else X_train[0],
            y = y_train if multiple_output else y_train[0],
            epochs = self.__config["max_epoch"],
            batch_size = self.__config["batch_size"],
            validation_data = (X_val if multiple_input else X_val[0], y_val if multiple_output else y_val[0]),
            callbacks = callbacks,
            sample_weight = sample_weight if not self.__config["regression"] else None,
            verbose = self.__config["verbose"],
        )
        print()
        
        if test:
            log_file = open(path.join(self.__config["logs_dir"], f"log_{self.__config["model_name"]}_{session_id}.txt"), "w")
            sys.stdout = Tee(log_file)
            y_pred = model.predict(X_test if multiple_input else X_test[0], batch_size=self.__config["max_epoch"], verbose=0)
            if multiple_output:
                y_pred = y_pred[main_output]

            t = sum(cb_timer.logs)
            print(f"SUMMARY AND TEST RESULTS\n")
            print(f" | Total train time: {convert_seconds(t)}")
            print(f" | Total epochs: {len(cb_timer.logs)}\n")
            if self.__config["regression"]:
                self.default_regression_test_function(y_test[main_output], y_pred, **test_function_options)
            else:
                self.default_classification_test_function(argmax(y_test[main_output]), argmax(y_pred), labels=test_labels, binary_classification=self.__config["binary_classification"], **test_function_options)
            Tee.reset()
            
        return weights_path