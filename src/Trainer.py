from NeuralNetwork import Network
from Metrics import Accuracy
import numpy as np
import sys


class SGDTrainer:
    # batchSize
    # learningRate
    # epochs
    # shuffle
    # sgdFlavor

    @staticmethod
    def optimize(network: Network, x, y, epochs=5, batch_size=10, alpha=0.03, validation_split=0.1,
                 metrics=[Accuracy()]):
        error_history = []
        metrics_history = []

        # split the data into train set and validation set
        data = split_data(x, y, validation_split, test_set_name="validation")

        data_valid = {
            "x": data["validation"]["x"],
            "y": data["validation"]["y"]
        }

        num_train_examples = len(data["train"]["y"])

        for epoch in range(epochs):
            i = 0

            # Shuffle train data
            indices = np.arange(num_train_examples)
            np.random.shuffle(indices)
            data_train = {
                "x": data["train"]["x"][indices],
                "y": data["train"]["y"][indices]
            }

            while i < num_train_examples:
                data_batch = {
                    "x": data_train["x"][i: i + batch_size],
                    "y": data_train["y"][i: i + batch_size]
                }
                i += batch_size

                # forward propagation
                tensors = network.forward(data_batch)

                # error
                error_history.append(tensors[-1][0].elements)

                # backward propagation
                deltas = network.backprop(tensors)
                deltas.reverse()

                # parameter update
                for layer, tensorlist in zip(network.layers, deltas):
                    if tensorlist is not None:
                        for tensor in tensorlist:
                            tensor.elements = alpha * tensor.elements
                        layer.update_parameter(tensorlist)

            # Evaluate with validation set
            epoch_metrics = [None] * len(metrics)
            prediction = network.predict(data_valid).elements
            for i in range(len(metrics)):
                epoch_metrics[i] = metrics[i].score(prediction, data_valid["y"])
            metrics_history.append(epoch_metrics)

            # Print progress update
            sys.stdout.write(
                '\r' + "Epoch %i / " % (epoch + 1) + str(epochs) + " Metrics: " + str(epoch_metrics))
            sys.stdout.flush()

        return error_history, metrics_history


def split_data(x, y, test_size_percent, test_set_name="test"):
    """
    Splits the data into training set and test set.
    :param x:
    :param y:
    :param test_size_percent:
    :param test_set_name:
    :return:
    """
    # Split the data randomly
    num_examples = len(y)
    indices = np.arange(num_examples)
    # np.random.seed = 42
    np.random.shuffle(indices)
    train_indices = indices[: int(num_examples * (1 - test_size_percent))]
    test_indices = indices[int(num_examples * (1 - test_size_percent)):]
    data = {
        "train": {
            "x": x[train_indices],
            "y": y[train_indices]
        },
        test_set_name: {
            "x": x[test_indices],
            "y": y[test_indices]
        }
    }
    return data
