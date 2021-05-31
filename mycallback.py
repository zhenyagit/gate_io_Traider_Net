from tensorflow.keras.callbacks import Callback
import json

class MyTrainingCallback(Callback):
    def __init__(self, save_directory):
        print("Callback is work")
        self.save_directory = save_directory
        self.full_data =\
            {
                "epoch" : []
            }
        self.epoch_data =\
            {
                "batch": []
            }
        self.batch_data = \
            {
                "accuracy": [],
                "loss": []
            }

    def on_train_batch_end(self, batch, logs=None):
        self.batch_data["accuracy"].append(logs['accuracy'])
        self.batch_data["loss"].append(logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_data["batch"].append(self.batch_data)
        self.full_data["epoch"].append(self.epoch_data)
        with open(self.save_directory+"endepoch"+str(epoch)+".json", 'w') as outfile:
            json.dump(self.epoch_data, outfile)
        for key in self.batch_data:
            self.batch_data[key] = []

    def on_train_end(self, logs=None):
        with open(self.save_directory + "endtrain.json", 'w') as outfile:
            json.dump(self.full_data, outfile)
