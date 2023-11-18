import os
import pathlib
import sys

from pandas import read_csv
from CustomDataset import CustomDataSt as cus_dtst
import torch
import torch.nn as nn

from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

project_root = pathlib.Path().resolve()
dataDir = 'data'
testFile = 'test.csv'
trainFile = 'train.csv'
validationFile = 'validation.csv'
path_to_file = os.path.join(project_root, dataDir)


def train_model(train_dataset, batch_size, lrn_rt, model, num_l):
    # define the optimization
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lrn_rt)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(train_dl)

    loss_val = 0
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
        # clear the gradients
        optimizer.zero_grad()

        model.hidden_cell = (
        torch.zeros(num_l, len(inputs), model.hidden_layer_size, dtype=torch.double),
        torch.zeros(num_l, len(inputs), model.hidden_layer_size, dtype=torch.double))

        model.to(dtype=torch.float64)
        output = model(inputs)

        # calculate loss
        loss = criterion(output.reshape(len(targets), -1), targets.reshape(len(targets), -1))
        loss_val += loss.item()
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()

    return {"loss": loss_val/num_batches}


def evaluate_model(dataset, model, num_l):
    dl = DataLoader(dataset, batch_size=10, shuffle=True)
    loss_fn = nn.MSELoss()
    test_loss = 0
    num_batches = len(dl)

    with torch.no_grad():
        for X, y in dl:
            model.hidden_cell = (
            torch.zeros(num_l, len(X), model.hidden_layer_size, dtype=torch.double),
            torch.zeros(num_l, len(X), model.hidden_layer_size, dtype=torch.double))

            pred = model(X)
            actual = y.reshape(len(y), -1)

            test_loss += loss_fn(pred, actual).item()

    return {"loss": test_loss/num_batches}


class LSTM(nn.Module):
    def __init__(self, input_size=1, num_layers=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size, dtype=torch.double),
                            torch.zeros(num_layers, 1, self.hidden_layer_size, dtype=torch.double))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        predictions = self.linear(lstm_out)

        pred = predictions[:, -1, :]

        return pred


num_layer = -1
h_size = -1
model = None

if len(sys.argv) == 2:
    if sys.argv[1] == 'question-c':
        num_layer = 1
        h_size = 64
    elif sys.argv[1] == 'question-d':
        num_layer = 2
        h_size = 64
    elif sys.argv[1] == 'custom':
        num_layer = int(input("\nEnter the number of layers for the network: "))
        h_size = int(input("Enter the number of hidden unit per hiddent layer: "))
else:
    raise Exception('not enough arguments to run the script.\n'
                    'the script should be provided with a mode, network structure, to run.'
                    '\nrun info: python lstmregr.py [question-c/question-d/custom]')

hyper_parameters = {"epoch": [30, 20, 25], "batch_size": [5, 10, 50],
                    "learning_rate": [0.05, 0.01, 0.005]}

max_accuracy = 99999999999999
best_at_epoch = -1
best_batch_size = -1
best_lrn_rt = -1
best_epochs = -1

test_df = read_csv(os.path.join(path_to_file, testFile))
train_df = read_csv(os.path.join(path_to_file, trainFile))
val_df = read_csv(os.path.join(path_to_file, validationFile))

train_dtst = cus_dtst(train_df)
test_dtst = cus_dtst(test_df)
val_dtst = cus_dtst(val_df)

fig, ax = plt.subplots(len(hyper_parameters["epoch"]), 1, sharey=True)

for it in range(len(hyper_parameters["epoch"])):
    epochs = hyper_parameters["epoch"][it]
    batch_size = hyper_parameters["batch_size"][it]
    lrn_rt = hyper_parameters["learning_rate"][it]
    model = LSTM(num_layers=num_layer, hidden_layer_size=h_size)

    print("\nRunning training with following hyper parameters:")
    print("Number of epochs: " + str(epochs))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(lrn_rt))

    plot_X = np.arange(0, epochs, 1)
    train_loss = []
    val_loss = []

    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        print("\nTraining...")
        train_loss.append(train_model(train_dtst, batch_size, lrn_rt, model, num_layer)["loss"])
        if (t + 1) % 5 == 0:
            print(f"training loss at epoch {t + 1} : {train_loss[t]:>7f}")
        print("\nValidation")
        val_accuracy = evaluate_model(val_dtst, model, num_layer)
        val_loss.append(val_accuracy["loss"])
        if (t + 1) % 5 == 0:
            print(f"validation loss at epoch {t + 1} : {val_loss[t]:>7f}")
        if val_accuracy["loss"] < max_accuracy:
            max_accuracy = val_accuracy["loss"]
            best_lrn_rt = lrn_rt
            best_batch_size = batch_size
            best_at_epoch = t
            best_epochs = epochs
            torch.save(model.state_dict(), 'weights_only.pth')

    if len(hyper_parameters["epoch"]) == 1:
        ax.plot(plot_X, train_loss, color='red',
                label="train_loss: ep=" + str(epochs) + ", b_size=" + str(
                    batch_size) + ", lrn_rt=" + str(lrn_rt))
        ax.plot(plot_X, val_loss, color='green',
                label="val_loss: ep=" + str(epochs) + ", b_size=" + str(
                    batch_size) + ", lrn_rt=" + str(lrn_rt))
        ax.legend(loc='upper right')
    else:
        ax[it].plot(plot_X, train_loss, color='red',
                    label="train_loss: ep=" + str(epochs) + ", b_size=" + str(
                        batch_size) + ", lrn_rt=" + str(lrn_rt))
        ax[it].plot(plot_X, val_loss, color='green',
                    label="val_loss: ep=" + str(epochs) + ", b_size=" + str(
                        batch_size) + ", lrn_rt=" + str(lrn_rt))
        ax[it].legend(loc='upper right')

plt.show()
print("\nHyper parameters after tuning:")
print("Number of epochs:" + str(best_epochs))
print("Learning rate:" + str(best_lrn_rt))
print("Batch size:" + str(best_batch_size))

# loading the state_dict
model_new = LSTM(num_layers=num_layer, hidden_layer_size=h_size)
model_new.to(dtype=torch.double)
model_new.load_state_dict(torch.load('weights_only.pth'))

print("\nTesting on the best model...")
print("\nloss for train dataset:")
result = evaluate_model(train_dtst, model_new, num_layer)
print(f"{result['loss']:>7f}")
print("\nloss for validation dataset:")
result = evaluate_model(val_dtst, model_new, num_layer)
print(f"{result['loss']:>7f}")
print("\nloss for test dataset:")
result = evaluate_model(test_dtst, model_new, num_layer)
print(f"{result['loss']:>7f}")
print("Done!")
