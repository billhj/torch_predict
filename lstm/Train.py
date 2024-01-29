import torch
import torch.nn as nn
import SequentialModels
import DataProcess
import numpy as np
import pandas as pd
from torch.autograd import Variable
import sys
import Logger
sys.stdout = Logger.Logger('a.log', sys.stdout)
sys.stderr = Logger.Logger('a.log_file', sys.stderr)
def print_train(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


data_raw = pd.read_excel('2019.xlsx')
data_load = data_raw.iloc[:,1:].values

#minmaxmodel, data_normalized = DataProcess.dataNormalization(data_load, -1, 1)
#input_sequences = data_normalized[:,:-1]
#output_sequences = data_normalized[:,-1]

input_sequences = data_load[:,:-1]
output_sequences = data_load[:,-1]

XXX,YYY = DataProcess.split_sequences(input_sequences, output_sequences, 10, 2, 0)

mydevice = torch.device("cuda:0")
if torch.cuda.is_available():
    mydevice = torch.device("cuda:0")
    print("Running on GPU")
else:
    mydevice = torch.device("cpu")
    print("Running on CPU")

X_tensors = Variable(torch.Tensor(XXX).to(mydevice))
Y_tensors = Variable(torch.Tensor(YYY).to(mydevice))
SequentialModels.trainLSTMModel(X_tensors[:-200],Y_tensors[:-200], X_tensors[-200:],Y_tensors[-200:])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_train('Train')