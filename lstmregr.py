import os
import pathlib
import sys
import warnings

import numpy
from datetime import date
import calendar
from datetime import timedelta

import torch
import torch.nn as nn

from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from CustomDataset import split_data_to_dataframes, CustomDataSt


class LSTM(nn.Module):
    """
    Represents the RNN model.
    Exposes methods to train and evaluate the model with train, test, validation datasets
    Also, provides two ways to predict future values.

    General flow:
        Initialize, load the data, train the model, optionally evaluate the model,
        predict using the model.
    """
    train_dtst = None
    val_dtst = None
    test_dtst = None
    window = None
    train_df = None
    test_df = None
    val_df = None
    # only uses cache for predicting with just year and month because the predictions are not going
    # to change between consecutive calls with same year, month values.
    cache_results = dict()

    def __init__(self, input_size: int = 1, num_layers: int = 1, hidden_layer_size: int = 64,
                 output_size: int = 1, window: int = 15):
        """
        Initializes the model.

        :param input_size: number of expected features in a single instance.
        :param num_layers: number of recurrent layers.
        :param hidden_layer_size: size of the features in the hidden layers.
        :param output_size: size of the number predictions expected from the model.
        :param window: time steps to unravel for a single prediction.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.window = window
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size, dtype=torch.double),
                            torch.zeros(num_layers, 1, self.hidden_layer_size, dtype=torch.double))

    def forward(self, input_seq):
        """
        does a single forward pass through the model
        :param input_seq: sequence to push through the model.
        :return: prediction for the input sequence
        """
        if not self._check_if_data_is_loaded():
            warnings.warn("Setup the data by calling load_data method.")
            return

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out)
        pred = predictions[:, -1, :]

        return pred

    def _check_if_data_is_loaded(self) -> bool:
        return not (self.train_dtst is None or self.test_dtst is None or self.val_dtst is None)

    def load_data(self, path_to_data_file: str, split_factor: float = 0.1):
        """
        loads the data in the data file into dataframes and CustomDataSt objects
        (train, test, validation).

        :param path_to_data_file: path to the data to load.
        :param split_factor: the percentage of data that should be in test and validation set each.
        :return: nothing. exceptions will be raised if the errors while splitting the data according
            to the requested split percentage.
        """
        try:
            (self.train_df, self.val_df, self.test_df) = split_data_to_dataframes(
                path_to_data_file, split_factor)
            window = self.window
            print("Training data stat:\n")
            self.train_dtst = CustomDataSt(self.train_df, window=window)
            print("\nValidation data stat:\n")
            self.val_dtst = CustomDataSt(self.val_df, self.train_dtst.normalizer, window=window)
            print("\nTest data stat:\n")
            self.test_dtst = CustomDataSt(self.test_df, self.train_dtst.normalizer, window=window)
        except Exception as exp:
            raise RuntimeError(f'''Encountered error while loading the dataset.\nError details:\n
{str(exp)}''')

    def train_model(self, batch_size: int, lrn_rt: float) -> dict:
        """
        Uses the forward function and trains the model with training dataset.
        Should load data before training the model.

        :param batch_size: number of batches to pass through the model in a single forward pass.
        :param lrn_rt: learning rate for the backward propagation.
        :return: loss per batch
        """
        if not self._check_if_data_is_loaded():
            warnings.warn("Setup the data by calling load_data method.")
            return
        # define the optimization
        criterion = nn.MSELoss()
        optimizer = SGD(self.parameters(), lr=lrn_rt)
        train_dl = DataLoader(self.train_dtst, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator())
        num_batches = len(train_dl)

        loss_val = 0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()

            self.hidden_cell = (
                torch.zeros(self.num_layers, len(inputs), self.hidden_layer_size,
                            dtype=torch.double),
                torch.zeros(self.num_layers, len(inputs), self.hidden_layer_size,
                            dtype=torch.double))

            self.to(dtype=torch.float64)
            output = self(inputs)

            # calculate loss
            loss = criterion(output.reshape(len(targets), -1), targets.reshape(len(targets), -1))
            loss_val += loss.item()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        return {"loss": loss_val / num_batches}

    def evaluate_model(self, validation: bool = False, batch_size: int = 10) -> dict:
        """
        Evaluates the trained model with the test or validation dataset.

        :param validation: (bool) if False, uses the test set to evaluate the model.
        :param batch_size: number of batches to pass through the model in a single forward pass.
        :return: loss per batch
        """
        if not self._check_if_data_is_loaded():
            warnings.warn("Setup the data by calling load_data method.")
            return
        if validation:
            dl = DataLoader(self.val_dtst, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator())
        else:
            dl = DataLoader(self.test_dtst, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator())

        loss_fn = nn.MSELoss()
        test_loss = 0

        with torch.no_grad():
            for X, y in dl:
                self.hidden_cell = (
                    torch.zeros(self.num_layers, len(X), self.hidden_layer_size,
                                dtype=torch.double),
                    torch.zeros(self.num_layers, len(X), self.hidden_layer_size,
                                dtype=torch.double))

                pred = self(X)
                actual = y.reshape(len(y), -1)

                test_loss += loss_fn(pred, actual).item()

        return {"loss": test_loss / len(dl)}

    def predict_for_month(self, month: int = 1, year: int = 2023, show_plot: bool = False) -> dict:
        """
        Makes the predictions for the requested month and date.

        :param month: month for which the prediction should be made.
        :param year: year for which the prediction should be made.
        :param show_plot: whether to plot the predictions in a graph.

        :return:
            on success, dictionary with keys representing the dates for which the predictions
            were made and values representing predicted values for the date.
            on error during prediction, raises exceptions.
        """
        if not self._check_if_data_is_loaded():
            warnings.warn("Setup the data by calling load_data method.")
            return
        if year < 2023 or year > 2024:
            raise NotImplementedError("The prediction works only for future and not past 2024.")
        if month < 1 or month > 12:
            raise RuntimeError("Not a valid month. Value should be between 1-12")
        try:
            X = np.array([self.train_df[-self.window:]["Receipt_Count"]])
            X = self.train_dtst.normalizer.transform(X.reshape(-1, 1))

            req_yy_mm = f"{year}-{month}"
            if self.cache_results.get(req_yy_mm) is not None:
                # for repeated requests for the same yy and mm, the predictions are not going to
                # change so using a cache.
                self._plot_results(self.cache_results[req_yy_mm])
                return self.cache_results[req_yy_mm]

            data_end_date = date(2022, 12, 31)
            req_start_date = date(year, month, 1)
            num_predictions = int(((req_start_date - data_end_date) \
                                   + timedelta(days=calendar.monthrange(year, month)[1] - 1)).days)

            curr_date = date(2023, 1, 1)
            while num_predictions != 0:
                with torch.no_grad():
                    self.hidden_cell = (
                        torch.zeros(self.num_layers, 1, self.hidden_layer_size, dtype=torch.double),
                        torch.zeros(self.num_layers, 1, self.hidden_layer_size, dtype=torch.double))
                    pred = self(torch.from_numpy(X.reshape(1, self.window, 1)))

                    X = numpy.append(X[1:], pred)
                    yy_mm = f"{curr_date.year}-{curr_date.month}"
                    yy_mm_dd = f"{yy_mm}-{curr_date.day}"
                    unnorm_val = self.train_dtst.normalizer.inverse_transform(
                        pred.numpy().reshape(-1, 1))[0][0]
                    if self.cache_results.get(yy_mm) is None:
                        self.cache_results[yy_mm] = {f"{yy_mm_dd}": unnorm_val}
                    else:
                        self.cache_results.get(yy_mm)[yy_mm_dd] = unnorm_val

                    curr_date += timedelta(days=1)
                    num_predictions -= 1
            if show_plot:
                self._plot_results(self.cache_results[req_yy_mm])
            return self.cache_results[req_yy_mm]

        except Exception as exp:
            raise RuntimeError(f'''Error while trying to predict for the provided month {month}, 
            year {year}.\nError details:\n\n{str(exp)}''')

    def predict_for_month_with_prev(self, prev_data: list, month: int = 1, year: int = 2023,
                                    show_plot: bool = False) -> dict:
        """
        Makes the predictions for the requested month and date using the provided previous data
        points.

        :param prev_data: (list of ints) previous data points to make the prediction,
                    length should match the window initialized in the model.
        :param month: month for which the prediction should be made.
        :param year: year for which the prediction should be made.
        :param show_plot: whether to plot the predictions in a graph.

        :return:
            on success, dictionary with keys representing the dates for which the predictions
            were made and values representing predicted values for the date.
            on error during prediction, raises exceptions.
        """
        if not self._check_if_data_is_loaded():
            warnings.warn("Setup the data by calling load_data method.")
            return
        if len(prev_data) != self.window:
            raise RuntimeError(f'''The amount of data provided should match the sequence length \
expected by the model {self.window}''')
        if month < 1 or month > 12:
            raise RuntimeError("Not a valid month. Value should be between 1-12")
        try:
            X = np.array(prev_data)
            X = self.train_dtst.normalizer.transform(X.reshape(-1, 1))

            req_start_date = date(year, month, 1)
            num_predictions = int(timedelta(days=calendar.monthrange(year, month)[1]).days)
            predictions = dict()

            curr_date = req_start_date
            while num_predictions != 0:
                with torch.no_grad():
                    self.hidden_cell = (
                        torch.zeros(self.num_layers, 1, self.hidden_layer_size, dtype=torch.double),
                        torch.zeros(self.num_layers, 1, self.hidden_layer_size, dtype=torch.double))
                    pred = self(torch.from_numpy(X.reshape(1, self.window, 1)))

                    X = numpy.append(X[1:], pred)
                    yy_mm = f"{curr_date.year}-{curr_date.month}"
                    yy_mm_dd = f"{yy_mm}-{curr_date.day}"
                    unnorm_val = self.train_dtst.normalizer.inverse_transform(
                        pred.numpy().reshape(-1, 1))[0][0]
                    predictions[yy_mm_dd] = unnorm_val

                    curr_date += timedelta(days=1)
                    num_predictions -= 1
            if show_plot:
                self._plot_results(predictions)
            return predictions

        except Exception as exp:
            raise RuntimeError(f'''Error while trying to predict for the provided month {month}, 
            year {year}.\nError details:\n\n{str(exp)}''')

    def _plot_results(self, results: dict, show_plot: bool = False):
        plt.plot(results.keys(), [round(val, 4) for val in results.values()])
        plt.xlabel("dates")
        plt.ylabel("number of receipts")
        plt.xticks(rotation=90)
        if show_plot:
            plt.show()
        else:
            plt.savefig("prediction_results.png")


def tune_model(rnn_model: LSTM, path_to_data_file: str, show_plot: bool = False):
    """
    Tunes the provided model with different hyper parameters, at the end saves the weights of the
    model in a state file named:
        weights_only_{rnn_model.num_layers}_{rnn_model.hidden_layer_size}_{rnn_model.window}.pth

    :param rnn_model: instance of LSTM class to tune / train.
    :param path_to_data_file: path to the data file.
    :param show_plot: whether to show the plot or not.

    :return: nothing.
    """
    rnn_model.load_data(path_to_data_file)

    hyper_parameters = {"epoch": [30, 20, 25], "batch_size": [5, 10, 50],
                        "learning_rate": [0.05, 0.001, 0.005]}

    max_accuracy = 99999999999999
    best_batch_size = -1
    best_lrn_rt = -1
    best_epochs = -1
    fig, ax = plt.subplots(len(hyper_parameters["epoch"]), 1, sharey=True)

    for it in range(len(hyper_parameters["epoch"])):
        epochs = hyper_parameters["epoch"][it]
        batch_size = hyper_parameters["batch_size"][it]
        lrn_rt = hyper_parameters["learning_rate"][it]

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
            train_loss.append(rnn_model.train_model(batch_size, lrn_rt)["loss"])
            if (t + 1) % 5 == 0:
                print(f"training loss at epoch {t + 1} : {train_loss[t]:>7f}")
            print("\nValidation")
            val_accuracy = rnn_model.evaluate_model(validation=True)
            val_loss.append(val_accuracy["loss"])
            if (t + 1) % 5 == 0:
                print(f"validation loss at epoch {t + 1} : {val_loss[t]:>7f}")
            if val_accuracy["loss"] < max_accuracy:
                max_accuracy = val_accuracy["loss"]
                best_lrn_rt = lrn_rt
                best_batch_size = batch_size
                best_epochs = epochs
                torch.save(rnn_model.state_dict(),
                           f"weights_only_{rnn_model.num_layers}_{rnn_model.hidden_layer_size}_{rnn_model.window}.pth")

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

    if show_plot:
        plt.show()
    else:
        plt.savefig("model_training.png")
    print("\nHyper parameters after tuning:")
    print("Number of epochs:" + str(best_epochs))
    print("Learning rate:" + str(best_lrn_rt))
    print("Batch size:" + str(best_batch_size))


def run_evaluations(rnn_model: LSTM, path_to_data_file: str):
    """
    Runs evaluation on the provided model using the test and validation model.
    This method loads the best weights saved during the tunng phase.

    :param rnn_model: instance of LSTM.
    :param path_to_data_file: path to the data file.
    :return: nothing.
            exceptions when there are errors during evaluation with testing / validation set.
    """
    try:
        rnn_model.to(dtype=torch.double)
        rnn_model.load_state_dict(torch.load(
            f"weights_only_{rnn_model.num_layers}_{rnn_model.hidden_layer_size}_{rnn_model.window}.pth"))
        rnn_model.load_data(path_to_data_file)
        print("\nTesting on the best model...")
        print("\nloss for validation dataset:")
        result = rnn_model.evaluate_model(validation=True)
        print(f"{result['loss']:>7f}")
        print("\nloss for test dataset:")
        result = rnn_model.evaluate_model()
        print(f"{result['loss']:>7f}")
        print("Done!")
    except Exception as exp:
        raise RuntimeError(f"Error while loading the model for inference:\ndetails:\n\n{str(exp)}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == 'custom':
            num_layer = int(input("\nEnter the number of layers for the network: "))
            h_size = int(input("Enter the number of hidden unit per hidden layer: "))
        else:
            raise Exception(f'''The argument to run the script is not correct.\n\
            the script should be provided with a mode to run, it supports 2 options.\n\
            \nrun info: python lstmregr.py [default|custom])''')
    else:
        num_layer = 2
        h_size = 16

    model = LSTM(num_layers=num_layer, hidden_layer_size=h_size)

    project_root = pathlib.Path().resolve()
    dataDir = 'data'
    dataFile = 'data_daily.csv'
    path_to_file = os.path.join(project_root, dataDir, dataFile)

    tune_model(model, path_to_file)

    model_new = LSTM(num_layers=num_layer, hidden_layer_size=h_size)
    run_evaluations(model_new, path_to_file)
