import os
from matplotlib import pyplot as plt
import json
import numpy as np
import random
from LSTM_net import OnlyNet

class CandlestickDataCollectorLSTM:
    def __init__(self, data_folder="./10s/", model_folder="./models/fitted10s/", inp_len=180, out_len=1):
        self.raw_data = None
        self.out_len = out_len
        self.inp_len = inp_len
        self.data_folder = data_folder
        self.get_full_from_folder()
        self.draw_array_candlestick(self.ret_data_to_LSTM(time=1622548887))
        self.model_folder = model_folder
        self.net = OnlyNet(self.model_folder)
        # self.draw_momentum_data()

    def ret_raw_data(self):
        return self.raw_data

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
        self.raw_data = full_array

    def ret_data_to_LSTM(self, time, array_full=None):
        index = 0
        if array_full == None:
            array_full = self.raw_data
        func = np.vectorize(lambda x: abs(x - time))
        delta = func(np.array(array_full)[:, 0])
        index = np.argmin(delta)
        return array_full[index - self.inp_len:index]

    def prediction_LSTM(self, time):
        x = self.ret_data_to_LSTM(time)
        return self.net.prediction(x)

    @staticmethod
    def convert_to_LSTM(array, data_to_take=2):
        return np.array(array)[:, data_to_take]

    def draw_array_candlestick(self, array=None):
        if array is None:
            array = self.raw_data
        plt.plot(np.array(array)[:, 0], np.array(array)[:, 2], marker='8', color='b')
        plt.show()



class MomentumData:
    def __init__(self, time, asks, bids, price):
        self.time = time
        self.asks = np.array(asks)
        self.bids = np.array(bids)
        self.price = price
        self.asks_div, self.asks_sum = self.ret_div_sum_data(self.asks)
        self.bids_div, self.bids_sum = self.ret_div_sum_data(self.bids)
        self.asks_mean, self.asks_disp = self.ret_mean_disp_data(self.asks, self.asks_div)
        self.bids_mean, self.bids_disp = self.ret_mean_disp_data(self.bids, self.bids_div)

    def ret_div_sum_data(self, array):
        copy_val = self.asks[:, 1]
        summ = np.sum(copy_val)
        return copy_val/summ, summ

    def ret_mean_disp_data(self, full_arr, array_div):
        mean = 0
        for i in range(len(array_div)):
            mean += full_arr[i][0] * array_div[i]
        disp = 0
        for i in range(len(array_div)):
            disp += full_arr[i][0] * full_arr[i][0] * array_div[i]
        disp = disp - mean * mean
        return mean, disp


class MomentumDataCollector:
    def __init__(self, data_folder="./momentum/"):
        self.data_folder = data_folder
        self.file_list = None
        self.get_file_list()

    def get_file_list(self):
        name_of_files = os.listdir(self.data_folder)
        temp = []
        for item in name_of_files:
            two = item.split("-")
            temp.append([float(two[0]), int(two[1][0:10])])
        temp = sorted(temp, key=lambda time: time[1])
        self.file_list = temp

    def ret_momentum(self, time, array=None):
        if array is None:
            array = self.file_list
        func = np.vectorize(lambda x: abs(x - time))
        delta = func(np.array(array)[:, 1])
        index = np.argmin(delta)
        file_name = str(array[index][0]) + "-" + str(array[index][1]) + ".json"
        raw_asks, raw_bids = self.ret_file_data(file_name)
        temp = []
        for item in raw_asks:
            temp.append([float(item[0]), float(item[1])])
        asks = temp
        temp = []
        for item in raw_bids:
            temp.append([float(item[0]), float(item[1])])
        bids = temp
        return MomentumData(np.array(array)[index, 1], asks, bids, np.array(array)[index, 0])

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
                 days_folder='./1d/', hours_folder='./1h/', minutes_folder='./5m/', model_folder="./models/fitted",
                 seconds_folder='./10s/'):
        self.start_time = 1622332800
        self.time_interval = time_interval
        self.money_USDT = money_USDT
        self.money_CRYP = money_CRYP
        self.momentum_folder = momentum_folder
        self.days_folder = days_folder
        self.hours_folder = hours_folder
        self.minutes_folder = minutes_folder
        self.seconds_folder = seconds_folder
        self.model_folder = model_folder
        # self.candle1d = CandlestickDataCollectorLSTM(data_folder=self.days_folder)
        self.candle1h = CandlestickDataCollectorLSTM(data_folder=self.hours_folder, model_folder=self.model_folder+self.hours_folder+"/")
        self.candle5m = CandlestickDataCollectorLSTM(data_folder=self.minutes_folder, model_folder=self.model_folder+self.minutes_folder+"/")
        self.candle10s = CandlestickDataCollectorLSTM(data_folder=self.seconds_folder, model_folder=self.model_folder+self.seconds_folder+"/")
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
        self.check_time_intervals_momentum()

    def reset(self):
        self.start_time = random.choice(self.time_can_use)
        self.now_time = self.start_time
        self.now_money_USDT = self.money_USDT
        self.now_money_CRYP = self.money_CRYP
        self.now_score = self.start_score
        observ = self.generate_observation()
        self.start_price = self.now_moment.price
        self.calculate_reward_score()
        self.orders = []
        self.success_orders = []
        return observ

    def buy(self, qtty=1):
        if self.now_money_USDT >= qtty:
            self.now_money_USDT -= qtty
        self.orders.append(Order(self.now_time, self.now_price, qtty, False))

    def sell(self, qtty=1):
        if self.now_money_CRYP*self.now_price >= qtty:
            self.now_money_CRYP -= qtty / self.now_price
        self.orders.append(Order(self.now_time, self.now_price, qtty, True))

    def step(self, action):
        self.check_orders()
        if action == 0:
            self.sell()
        if action == 1:
            self.buy()
        if action == 2:
            pass
        self.now_time += 1
        reward = self.calculate_reward_score()
        observation = self.generate_observation()
        done = self.check_done()
        info = None
        return observation, reward, done, info

    def check_done(self):
        return self.now_time > self.start_time + self.time_interval


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
        return out_observ

    def calculate_reward_score(self):
        temp_score = self.now_money_USDT + self.now_money_USDT * self.start_price + self.calculate_money_in_orders()
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
                self.success_orders.append(self.orders.pop(i))
                if self.orders[i].sell_by:
                    self.now_money_USDT += self.orders[i].qtty - self.orders[i].qtty * fee
                else:
                    self.now_money_CRYP += self.orders[i].qtty / self.now_price - \
                                           (self.orders[i].qtty / self.now_price) * fee
            i += 1

    def check_time_intervals_momentum(self):
        times = np.array(self.moment.file_list)[:, 1]
        for i in range(len(times) - self.time_interval-1):
            check = False
            for k in range(self.time_interval):
                if times[i+self.time_interval-k+1] - times[i+self.time_interval-k]> 8:
                    check = True
            if check is False:
                self.time_can_use.append(i+self.time_interval)

