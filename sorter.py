import os
import json
import numpy as np
import glob


class Sorter():
    def __init__(self, nametype, path_candelsticks, save_folder=None):
        self.folder_path = path_candelsticks
        self.nametype = nametype
        if save_folder == None:
            self.save_folder = os.path.join(self.folder_path, nametype)
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        self.all_data_dic = []

        self.load_full()

    def load_full(self):
        list_of_files = [f for f in glob.glob(self.folder_path + self.nametype + '-*')]
        for file in list_of_files:
            with open(file, 'r') as opened_file:
                data = json.load(opened_file)
            for lineindex in range(len(data)):
                if len(self.all_data_dic) == 0:
                    temp = []
                    temp.append(int(data[lineindex][0]))
                    for i in range(5):
                        temp.append(float(data[lineindex][i + 1]))
                    self.all_data_dic.append(temp)
                else:
                    if not int(data[lineindex][0]) in np.array(self.all_data_dic).transpose()[0]:
                        temp = []
                        temp.append(int(data[lineindex][0]))
                        for i in range(5):
                            temp.append(float(data[lineindex][i + 1]))
                        self.all_data_dic.append(temp)
        self.all_data_dic = sorted(self.all_data_dic, key=lambda time: time[0])
        with open(self.save_folder + '/all_in_one.json', 'w', encoding='utf-8') as file:
            json.dump(self.all_data_dic, file, ensure_ascii=False)


sor = Sorter("1d", "./canelsticks/")
