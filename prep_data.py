import pandas as pd
import os
import numpy as np
from keras.utils import to_categorical

train_signals_folder_path = "./UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/"
train_label_path = "./UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"
train_signal_names = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                      'body_gyro_x_train.txt', 'body_gyro_y_train.txt','body_gyro_z_train.txt']

test_signals_folder_path = "./UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals/"
test_label_path = "./UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt"
test_signal_names = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                    'body_gyro_x_test.txt', 'body_gyro_y_test.txt','body_gyro_z_test.txt']




def get_data(signals_folder_path, label_path, signal_names, signal_data_zeros):

    for i in range(len(signal_names)):
        path = os.path.join(signals_folder_path, signal_names[i])
        data = pd.read_csv(path, sep="\s+", header=None)
        signal_data_zeros[:,:,i] = data.to_numpy()
    
    signal_data = z_normalization(signal_data_zeros)

    data = pd.read_csv(label_path, sep="\s+", header=None)
    signal_labels = to_categorical(data.to_numpy()-1, num_classes=6)

    return signal_data, signal_labels

def z_normalization(data):
    mean = np.mean(data, axis=0)
    dev = np.std(data, axis=0)
    return (data - mean) / dev 


def get_train_data():
    signal_data = np.zeros((7352,128,6))
    return get_data(train_signals_folder_path, train_label_path, train_signal_names, signal_data)

def get_test_data():
    signal_data = np.zeros((2947,128,6))
    return get_data(test_signals_folder_path, test_label_path, test_signal_names, signal_data)

if __name__ == "__main__":
    data, labels = get_train_data()
    print("Train: ")
    print(data.shape)
    print(labels.shape)

    data, labels = get_test_data()
    print("Test: ")
    print(data.shape)
    print(labels.shape)

