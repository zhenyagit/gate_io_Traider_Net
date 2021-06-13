import os
from matplotlib import pyplot as plt
import json
import numpy as np
import random
from LSTM_net import OnlyNet
from line_profiler_pycharm import profile
from functools import  lru_cache

class CandlestickDataCollectorLSTM:
    def __init__(self, data_folder="./10s/", model_folder="./models/fitted10s/", index_file=None, inp_len=180, out_len=1):
        self.raw_data = None
        self.raw_data_times = None
        self.out_len = out_len
        self.inp_len = inp_len
        self.data_folder = data_folder
        self.get_full_from_folder()
        # self.draw_array_candlestick(self.ret_data_to_LSTM(time=1622548887))
        self.model_folder = model_folder
        self.net = OnlyNet(self.model_folder, index_file)
        self.time_prediction = None
        self.check_times_predictions()
        # self.draw_momentum_data()

    def ret_raw_data(self):
        return self.raw_data

    def check_times_predictions(self):
        if os.path.isfile(self.model_folder + "time_prediction.json"):
            with open(self.model_folder + "time_prediction.json", 'r') as file:
                self.time_prediction = np.array(json.load(file))

    def get_full_from_folder(self, path_to_folder=None):
        if path_to_folder is None:
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
        full_array = sorted(full_array, key=lambda time: time[0])
        self.raw_data = np.array(full_array)[:, 2]
        self.raw_data_times = np.array(full_array)[:, 0]

    def nearest_value(self, time):
        lenth = len(self.raw_data_times)
        lenth = round(lenth/2)
        x = lenth
        while lenth > 1:
            lenth = round(lenth / 2)
            if self.raw_data_times[x] == time:
                break
            if self.raw_data_times[x] > time:
                x = x - lenth
            else:
                x = x + lenth
        func = np.vectorize(lambda x: abs(x - time))
        delta = func(self.raw_data_times[x-5:x+5])
        index = np.argmin(delta)+x-5
        return index
#1702062464
#  12560801
    @lru_cache
    @profile
    def ret_data_to_LSTM(self, time, array_full=None):
        if array_full is None:
            array_full = self.raw_data
        index = self.nearest_value(time)
        ret_array = array_full[index - self.inp_len - 1:index]
        new_arr = ret_array[:-1] - ret_array[1:]
        return new_arr
#2136143616
# 496405504
    @profile
    def generate_full_to_LSTM(self, times):
        full_data = np.empty(shape=(len(times), len(self.ret_data_to_LSTM(times[0]))))
        for i in range(len(times)):
            full_data[i] = self.ret_data_to_LSTM(times[i])
        full_data = np.reshape(full_data, (full_data.shape[0], full_data.shape[1], 1))
        predictions = self.net.many_predictions(full_data)
        self.time_prediction = np.empty(shape=(2, len(times)))
        self.time_prediction[0] = np.array(times)
        self.time_prediction[1] = np.reshape(predictions, (len(times)))
        with open(self.model_folder + "time_prediction.json", 'w') as file:
            json.dump(self.time_prediction.tolist(), file)

    def prediction_LSTM(self, time):
        if self.time_prediction is None:
            x = self.ret_data_to_LSTM(time)
            return self.net.one_prediction(x)
        else:
            index = np.where(self.time_prediction[0] == time)
            try:
                temp = self.time_prediction[1, index[0][0]]
            except Exception:
                x = self.ret_data_to_LSTM(time)
                temp = self.net.one_prediction(x)
            return temp

    @staticmethod
    def convert_to_LSTM(array, data_to_take=2):
        return np.array(array)[:, data_to_take]

    def draw_array_candlestick(self, array=None):
        if array is None:
            array = self.raw_data
        plt.plot(self.raw_data_times, self.raw_data, marker='8', color='b')
        plt.show()


class Loger:
    def __init__(self):
        self.full_data = []

    def add(self, data):
        self.full_data.append(data)

    def write(self, time, path="./env_log/"):
        with open(path + "log-" + str(time) + ".json", 'w') as file:
            json.dump(self.full_data, file)


class MomentumData:
    @profile
    def __init__(self, time, asks, bids, price):
        self.time = time
        self.asks = asks
        self.bids = bids
        self.price = price
        self.asks_div, self.asks_sum = self.ret_div_sum_data()
        self.bids_div, self.bids_sum = self.ret_div_sum_data()
        self.asks_mean, self.asks_disp = self.ret_mean_disp_data(self.asks, self.asks_div)
        self.bids_mean, self.bids_disp = self.ret_mean_disp_data(self.bids, self.bids_div)

    def ret_div_sum_data(self):
        copy_val = self.asks[:, 1]
        summ = np.sum(copy_val)
        return copy_val/summ, summ

    def ret_mean_disp_data(self, full_arr, array_div):
        mean = np.sum(full_arr[:, 0] * array_div[:])
        disp = np.sum(full_arr[:, 0] * full_arr[:, 0] * array_div[:])
        disp = disp - mean * mean
        return mean, disp


class MomentumDataCollector:
    def __init__(self, data_folder="./momentum/"):
        self.data_folder = data_folder
        self.file_list = None
        self.file_names = None
        self.get_file_list()

    def get_file_list(self):
        self.file_names = os.listdir(self.data_folder)
        temp = []
        for item in self.file_names:
            two = item.split("-")
            temp.append([float(two[0]), int(two[1][0:10])])
        temp = sorted(temp, key=lambda time: time[1])
        self.file_list = np.array(temp)
#19281292
# 3580401

    @profile
    def ret_momentum(self, time, array=None):
        if array is None:
            array = self.file_list
        func = np.vectorize(lambda x: abs(x - time))
        delta = func(array[:, 1])
        index = np.argmin(delta)
        file_name = self.file_names[index]
        raw_asks, raw_bids = self.ret_file_data(file_name)
        asks = np.empty(shape=(len(raw_asks), 2))
        for i in range(len(raw_asks)):
            asks[i, 0] = float(raw_asks[i][0])
            asks[i, 1] = float(raw_asks[i][1])
        bids = np.empty(shape=(len(raw_asks), 2))
        for i in range(len(raw_bids)):
            bids[i, 0] = float(raw_bids[i][0])
            bids[i, 1] = float(raw_bids[i][1])
        return MomentumData(array[index, 1], asks, bids, array[index, 0])

    def ret_file_data(self, name_file):
        with open(self.data_folder + name_file, 'r') as file:
            data = json.load(file)
        return data['asks'], data['bids']

    def draw_momentum_data(self):
        name_of_files = os.listdir(self.data_folder)
        temp = []
        for item in name_of_files:
            two = item.split("-")
            temp.append([float(two[0]), int(two[1][0:10])])
        temp = sorted(temp, key=lambda time: time[1])
        plt.plot(np.array(temp)[:, 1], np.array(temp)[:, 0], marker='8', color='b')
        plt.show()


class Order:
    def __init__(self, time, price, qtty, sell_buy):
        self.time = time
        self.price = price
        self.sell_buy = sell_buy
        self.qtty = qtty
        self.success = False

    def check(self, now_price):
        if self.sell_buy is True:
            if now_price < self.price:
                self.success = True
        else:
            if now_price > self.price:
                self.success = True
        return self.success


class EnvironmentGateIO:
    def __init__(self, money_USDT, money_CRYP,  time_interval=60*60*2, momentum_folder="./momentum/",
                 days_folder='./1d/', hours_folder='./1h/', minutes_folder='./5m/', seconds_folder='./10s/',
                 model_folder1h="./models/fitted1h/", model_folder5m="./models/fitted5m/",
                 model_folder10s="./models/fitted10s/"):
        self.start_time = 1622332800
        self.time_interval = time_interval
        self.money_USDT = money_USDT
        self.money_CRYP = money_CRYP
        self.momentum_folder = momentum_folder
        self.days_folder = days_folder
        self.hours_folder = hours_folder
        self.minutes_folder = minutes_folder
        self.seconds_folder = seconds_folder
        self.model_folder1h = model_folder1h
        self.model_folder5m = model_folder5m
        self.model_folder10s = model_folder10s
        # self.candle1d = CandlestickDataCollectorLSTM(data_folder=self.days_folder)
        self.candle1h = CandlestickDataCollectorLSTM(data_folder=self.hours_folder, model_folder=self.model_folder1h, index_file=100)
        self.candle5m = CandlestickDataCollectorLSTM(data_folder=self.minutes_folder, model_folder=self.model_folder5m, index_file=100)
        self.candle10s = CandlestickDataCollectorLSTM(data_folder=self.seconds_folder, model_folder=self.model_folder10s, index_file=90)
        self.moment = MomentumDataCollector(data_folder=self.momentum_folder)
        self.now_money_USDT = None
        self.now_money_CRYP = None
        self.now_price = None
        self.now_time = None
        self.now_moment = None
        self.now_score = None
        self.start_price = None
        self.start_score = None
        self.orders = []
        self.success_orders = []
        self.time_can_use = []
        self.loger = None
        print("Start check intervals")
        self.check_time_intervals_momentum()
        print("Stop check intervals")

    @profile
    def reset(self):
        self.orders = []
        self.success_orders = []
        self.start_time = random.choice(self.time_can_use)
        self.now_time = self.start_time
        self.now_moment = self.moment.ret_momentum(self.now_time)
        self.now_price = self.now_moment.price
        self.start_price = self.now_moment.price
        self.now_money_USDT = self.money_USDT
        self.now_money_CRYP = self.money_CRYP
        self.now_score = 0
        self.calculate_reward_score()
        self.start_score = self.now_score
        print("Generating all predictions")
        self.candle1h.generate_full_to_LSTM(range(int(self.start_time), int(self.start_time) + self.time_interval+4))
        self.candle5m.generate_full_to_LSTM(range(int(self.start_time), int(self.start_time) + self.time_interval+4))
        self.candle10s.generate_full_to_LSTM(range(int(self.start_time), int(self.start_time) + self.time_interval+4))
        print("Done")
        observ = self.generate_observation()
        # if self.loger is None:
        #     self.loger = Loger()
        # else:
        #     self.loger.write(self.now_time)
        #     self.loger = Loger()
        return observ

    def buy(self, qtty=1):
        if self.now_money_USDT >= qtty:
            self.now_money_USDT -= qtty
            self.orders.append(Order(self.now_time, self.now_price, qtty, False))

    def sell(self, qtty=1):
        if self.now_money_CRYP*self.now_price >= qtty:
            self.now_money_CRYP -= qtty / self.now_price
            self.orders.append(Order(self.now_time, self.now_price, qtty, True))
# 6516762
#  949263
    @profile
    def step(self, action):
        self.check_orders()
        if action == 0:
            self.sell()
        if action == 1:
            self.buy()
        if action == 2:
            pass
        self.now_time += 3
        if self.now_time > self.start_time + self.time_interval:
            self.now_time = self.start_time + self.time_interval
        reward = self.calculate_reward_score()
        observation = self.generate_observation()
        done = self.check_done()
        info = self.calculate_progress()
        # self.loger.add([action, observation.tolist(), reward, info])
        return observation, reward, done, info

    def check_done(self):
        return self.now_time >= self.start_time + self.time_interval

    def calculate_progress(self):
        progress = (self.now_time - self.start_time) / self.time_interval * 100
        return progress
# 13075640
#  1911762
    @profile
    def generate_observation(self):
        self.now_moment = self.moment.ret_momentum(self.now_time)
        self.now_price = self.now_moment.price
        out_observ = []
        summ_in_order_usdt = 0
        summ_in_order_cryp = 0
        for order in self.orders:
            if order.sell_buy is False:
                summ_in_order_usdt += order.qtty
            else:
                summ_in_order_cryp += order.qtty / order.price * self.now_price
        money = (self.now_money_USDT+summ_in_order_usdt) / (self.now_money_USDT + summ_in_order_usdt +
                                                            self.now_money_CRYP * self.now_price + summ_in_order_cryp)
        out_observ.append(self.candle1h.prediction_LSTM(time=self.now_time))
        out_observ.append(self.candle5m.prediction_LSTM(time=self.now_time))
        out_observ.append(self.candle10s.prediction_LSTM(time=self.now_time))
        out_observ.append(self.now_score)
        out_observ.append(money)
        out_observ.append(self.now_moment.asks_sum)
        out_observ.append(self.now_moment.asks_mean)
        out_observ.append(self.now_moment.asks_disp)
        out_observ.append(self.now_moment.bids_sum)
        out_observ.append(self.now_moment.bids_mean)
        out_observ.append(self.now_moment.bids_disp)
        out_observ.append(np.sin(self.now_time % (60*60*24)*np.pi*2/(60*60*24)))
        out_observ.append(self.now_moment.price)
        # [LSTM1h,LSTM5m,LSTM10s,score,money,asks*3,bids*3,time,price]
        return np.array(out_observ)

    def calculate_reward_score(self):
        temp_score = self.now_money_USDT + self.now_money_CRYP * self.start_price + self.calculate_money_in_orders()
        reward = temp_score - self.now_score
        self.now_score = temp_score
        return reward

    def calculate_money_in_orders(self):
        summ = 0
        for order in self.orders:
            if order.sell_buy is False:
                summ += order.qtty
            else:
                summ += order.qtty / order.price * self.now_price
        return summ

    def check_orders(self, fee=0.0018):
        i = 0
        while i < len(self.orders):
            if self.orders[i].check(self.now_price):
                if self.orders[i].sell_buy:
                    self.now_money_USDT += self.orders[i].qtty - self.orders[i].qtty * fee
                else:
                    self.now_money_CRYP += self.orders[i].qtty / self.now_price - \
                                           (self.orders[i].qtty / self.now_price) * fee
                self.success_orders.append(self.orders.pop(i))
                i -= 1
            i += 1

    def check_time_intervals_momentum(self):
        if os.path.isfile("time_can_use.json"):
            print("Warning!!! Loading old times")
            with open("time_can_use.json", 'r') as file:
                self.time_can_use = list(json.load(file))
        else:
            print("Warning!!! Processing new times")
            times = np.array(self.moment.file_list)[:, 1]
            for i in range(len(times) - self.time_interval-1):
                check = False
                for k in range(self.time_interval):
                    if times[i+self.time_interval-k+1] - times[i+self.time_interval-k]> 8:
                        check = True
                if check is False:
                    self.time_can_use.append(times[i+self.time_interval])
            with open("time_can_use.json", 'w') as file:
                json.dump(self.time_can_use, file)
        print("We can use ", len(self.time_can_use), " intervals")

