import pandas as pd
import os
import numpy as np
from keras.utils import to_categorical

signals_folder_path = "./UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/"
label_path = "./UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"

signal_names = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                'body_gyro_x_train.txt', 'body_gyro_y_train.txt','body_gyro_z_train.txt']

def get_data():
    signal_data = np.zeros((7352,128,6))

    for i in range(len(signal_names)):
        path = os.path.join(signals_folder_path, signal_names[i])
        data = pd.read_csv(path, sep="\s+", header=None)
        signal_data[:,:,i] = data.to_numpy()
    
    signal_data = z_normalization(signal_data)

    data = pd.read_csv(label_path, sep="\s+", header=None)
    signal_labels = to_categorical(data.to_numpy()-1, num_classes=6)

    return signal_data, signal_labels

def z_normalization(data):
    mean = np.mean(data, axis=0)
    dev = np.std(data, axis=0)
    return (data - mean) / dev 

if __name__ == "__main__":
    data, labels = get_data()
    print(data.shape)
    print(labels.shape)

