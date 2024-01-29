import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

import Evaluation

mydevice = torch.device("cuda:0")
if torch.cuda.is_available():
    mydevice = torch.device("cuda:0")
    print("Running on GPU")
else:
    mydevice = torch.device("cpu")
    print("Running on CPU")

#train loop function
def training_loop(n_epochs, model, optimiser, loss_fn, X_train, y_train, X_test, y_test):
    for epoch in range(n_epochs):
        model.train()
        optimiser.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(mydevice),
                             torch.zeros(1, 1, model.hidden_size).to(mydevice))
        outputs = model.forward(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimiser.step()

        model.eval()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(mydevice),
                             torch.zeros(1, 1, model.hidden_size).to(mydevice))
        test_preds = model(X_test)
        test_loss = loss_fn(test_preds, y_test)

        if (epoch + 1) % 5 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item()))

        if (epoch + 1) % 50 == 0:
            model.eval()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(mydevice),
                                 torch.zeros(1, 1, model.hidden_size).to(mydevice))
            train_preds = model(X_train)
            evalTrain = Evaluation.evalMetrics(y_train.cpu().detach().numpy(), train_preds.cpu().detach().numpy())
            print(f"End of {epoch}, train data mae {evalTrain[0]},  rmse {evalTrain[2]}, mape {evalTrain[3]}")

    model.eval()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(mydevice),
                         torch.zeros(1, 1, model.hidden_size).to(mydevice))
    train_preds = model(X_train)
    evalTrain = Evaluation.evalMetrics(y_train.cpu().detach().numpy(), train_preds.cpu().detach().numpy())
    print(f"End of {epoch}, train data mae {evalTrain[0]}, rmse {evalTrain[2]}, mape {evalTrain[3]}")

    model.eval()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(mydevice),
                         torch.zeros(1, 1, model.hidden_size).to(mydevice))
    test_preds = model(X_test)
    evalTest = Evaluation.evalMetrics(y_test.cpu().detach().numpy(), test_preds.cpu().detach().numpy())
    print(f"End of {epoch}, test data mae {evalTest[0]}, rmse {evalTest[2]}, mape {evalTest[3]}")
"""
    plt.title('timing p')
    plt.ylabel('p')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)

    plt.plot(train_preds[-1].detach().numpy())
    plt.plot(y_train[-1].detach().numpy())
    plt.show()
"""

def loadModel(filename):
    model =torch.load(filename)
    model.eval()
    return model

class LSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=100, num_layers=1):
        # num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM
        # hidden_size (the size of the output from the last LSTM layer)
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(mydevice))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(mydevice))
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        predictions = self.relu(hn)
        predictions = self.fc_1(predictions)
        predictions = self.relu(predictions)
        predictions = self.fc_2(predictions)
        return predictions  #[:, -1, :]




def trainLSTMModel(X_train, y_train, X_test, y_test, epochs=1000, savefilename='mylstm20240125', hidden_size=100, num_layers=1):
    input_size = X_train.shape[-1]
    output_size = y_train.shape[-1]
    model = LSTM(input_size, output_size, hidden_size, num_layers).to(mydevice)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    training_loop(epochs, model, optimizer, loss_function, X_train, y_train, X_test, y_test)
    filename = savefilename+'.pth'
    torch.save(model, filename)
    print(f"save model into file: {filename}")


def evalLSTMModel(modelfilename, data):
    model = torch.load(modelfilename)
    model.eval()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                         torch.zeros(1, 1, model.hidden_size))
    test_preds = model(data)
    return test_preds

if __name__ == '__main__':
    print("SequentialModels")