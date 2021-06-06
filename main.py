import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, ReLU, Flatten
from tensorflow.keras.activations import linear
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from mycallback import MyTrainingCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import datetime

class LSTMTry():
    def __init__(self, default_model_folder='./fitted/', default_data_folder='./data/', inp_len=180, out_len=1, timestap=10, testfile="10s-test.json"):
        self.inp_len = inp_len
        self.out_len = out_len
        self.test_file = testfile
        self.raw_data = None
        self.model = None
        self.model_callbacks = None
        self.train_data = None
        self.valid_data = None
        self.scaler = MinMaxScaler()
        self.data_folder = default_data_folder
        self.model_folder = default_model_folder
        self.get_full_from_folder(timestap=timestap)
        self.get_data_to_fit()
        self.setup_callback()

    def set_model_folder(self, path):
        self.model_folder = path

    def set_data_folder(self, path):
        self.data_folder = path

    def what_we_do(self):
        print("1 Load h5 and fit")
        print("2 Load h5 and predict")
        print("3 Load new and fit")
        print("4 Show len of input data")
        we_do = int(input())
        if we_do == 1:
            self.model_from_h5()
            self.draw_after_load()
            self.model_fit()
        if we_do == 2:
            self.model_from_h5()
            self.model_predict()
        if we_do == 3:
            self.load_model()
            self.draw_after_load()
            self.model_fit()
        if we_do == 4:
            print("Len of train data: ", np.array(self.train_data).shape)
            print("Len of valid data: ", np.array(self.valid_data).shape)
            self.what_we_do()

    def load_model(self):
        inp = Input((self.inp_len, 1))
        model = LSTM(180, return_sequences=True)(inp)  # 180*10/60 = 30 min
        model = Dropout(0.2)(model)
        model = LSTM(180, return_sequences=True)(model)
        model = Dropout(0.2)(model)
        model = LSTM(180, return_sequences=True)(model)
        model = Dropout(0.2)(model)
        model = LSTM(180, return_sequences=False)(model)

        # model = Flatten()(model)
        model = Dense(40, activation='linear')(model)
        model = ReLU()(model)
        model = Dense(self.out_len, activation='linear')(model)
        full_model = Model(inputs=inp, outputs=model)
        self.model = full_model
        self.model.summary()

    def get_full_from_folder(self, path_to_folder=None, timestap=10):
        if path_to_folder == None:
            path_to_folder = self.data_folder
        list_files = os.listdir(path_to_folder)
        list_files.sort()
        full_array = []
        for filename in list_files:
            with open(path_to_folder + str(filename), 'r') as file:
                data = json.load(file)
            for lineinfile in data:
                line = []
                for numberinline in lineinfile:
                    line.append(float(numberinline))
                full_array.append(line)
        full_arrays = []
        temp = []
        for indexinarray in range(len(full_array) - 1):
            if full_array[indexinarray + 1][0] - full_array[indexinarray][0] == timestap:
                temp.append(full_array[indexinarray])
            else:
                full_arrays.append(temp)
                temp = []
        if len(full_arrays) == 0 and len(full_array) != 0:
            full_arrays.append(full_array)
        self.raw_data = full_arrays

    def get_one_from_file(self, filename='1620588690.json'):
        full_array = []
        with open(filename, 'r') as file:
            data = json.load(file)
        for lineinfile in data:
            line = []
            for numberinline in lineinfile:
                line.append(float(numberinline))
            full_array.append(line)
        full_arrays = []
        temp = []
        for indexinarray in range(len(full_array) - 1):
            if full_array[indexinarray + 1][0] - full_array[indexinarray][0] == 10:
                temp.append(full_array[indexinarray])
            else:
                full_arrays.append(temp)
                temp = []
        if len(full_arrays) == 0 and len(full_array) != 0:            # check if all data in one array
            full_arrays.append(full_array)
        for i in range(len(full_arrays)):
            if len(full_arrays[i]) < self.inp_len + self.out_len:
                full_arrays.pop(i)
                i -= 1
        return full_arrays

    # 0 Unix timestamp in seconds
    # 1 Trading volume
    # 2 Close price
    # 3 Highest price
    # 4 Lowest price
    # 5 Open price

    def get_data_to_fit(self, array_full=None, data_to_take=2):
        if array_full == None:
            array_full = self.raw_data

        data_array = array_full[0]
        differences = []
        for arrayindex in range(len(data_array) - 1):
            differences.append((np.array(data_array[arrayindex]) - np.array(data_array[arrayindex + 1])).tolist())
        # differences = self.scaler.fit_transform(differences)
        data_array = np.array(differences)  # sorry
        train_data = []
        valid_data = []
        for arrayindex in range(self.inp_len, len(data_array) - self.out_len):
            train_data.append(data_array[arrayindex - self.inp_len:arrayindex, data_to_take])
            valid_data.append(data_array[arrayindex:arrayindex + self.out_len, data_to_take])
        self.train_data = train_data
        self.valid_data = valid_data

    def setup_callback(self, only_best=False):
        path_to_save = self.model_folder
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
            os.mkdir(path_to_save + "log/")
        callbacks = [MyTrainingCallback(path_to_save)]
        log_dir = path_to_save + "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)
        if not only_best:
            callbacks.append(
                ModelCheckpoint(path_to_save + 'gate_io-{epoch:02d}.hdf5', monitor='val_acc', save_best_only=only_best))
        else:
            callbacks.append(ModelCheckpoint(path_to_save + "best", monitor='val_accuracy', save_best_only=only_best))
        self.model_callbacks = callbacks

    def model_from_h5(self):
        list_model = os.listdir(self.model_folder)
        for i in range(len(list_model)):
            print(i, '\t', list_model[i])
        print("Enter index model to predict")
        model_index = int(input())
        self.model = load_model(self.model_folder + list_model[model_index])
        self.model.summary()

    def draw_after_load(self):
        plt.plot(range(0, self.inp_len), self.train_data[0], color='b', label='Train')
        plt.plot(range(self.inp_len, self.inp_len + self.out_len), self.valid_data[0], color='g', label='Valid')
        # plt.plot(range(self.inp_len, self.inp_len+self.out_len), self.train_data[self.inp_len][0:self.out_len],
        # color='r', label='Valid real')
        plt.show()
        # juasgdkljhagsdlkjighas;

    def model_predict(self):
        self.get_data_to_fit(self.get_one_from_file(filename=self.test_file))
        for i in range(40):
            number_to_try = i + 20
            x = np.array([self.train_data[number_to_try]])
            u = np.reshape(x, (x.shape[0], x.shape[1], 1))
            y = self.model.predict(u)
            plt.plot(range(170, self.inp_len), self.train_data[number_to_try][170:], color='b', label='Train')
            plt.plot(range(self.inp_len, self.inp_len + self.out_len),
                     self.train_data[self.inp_len + number_to_try][0:self.out_len], marker='8', color='r',
                     label='Valid real')
            plt.plot(range(self.inp_len, self.inp_len + self.out_len), y[0], color='g', marker='8', label='Valid')
            plt.show()

        print(y)

    def model_fit(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        t = np.array((self.train_data))
        v = np.array((self.valid_data))
        train_data = np.reshape(t, (t.shape[0], t.shape[1], 1))
        valid_data = np.reshape(v, (v.shape[0], v.shape[1], 1))
        print(train_data.shape)
        print(valid_data.shape)
        self.model.fit(train_data, valid_data, batch_size=32, epochs=50, callbacks=self.model_callbacks)

# net = LSTMTry(default_model_folder='./models/fitted10s/', default_data_folder='./canelsticks/10s/', inp_len=180, out_len=1)
net = LSTMTry(default_model_folder='./models/fitted5m/', default_data_folder='./canelsticks/5m/', inp_len=180, out_len=1, timestap=60*5)
net.what_we_do()
