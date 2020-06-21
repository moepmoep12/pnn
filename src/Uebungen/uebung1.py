from sklearn.datasets import fetch_openml
from Trainer import *
from Layer.Layer import *
from NeuralNetwork import *
from Scaler import *
from Layer.ActivationLayer import *
from Layer.Loss import *

class MnistInputLayer(InputLayer):
    def get_output_shape(self) -> Shape:
        return Shape([1, 784])

    def forward(self, raw_data):
        return [Tensor(elements=raw_data["x"])], [Tensor(elements=raw_data["y"])]


if __name__ == '__main__':
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    y = y.astype(int)
    # X = np.array([
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8]
    # ])
    # y = np.array([1, 0])
    y_one_hot = np.eye(y.max() + 1)[y]

    data = split_data(X, y_one_hot, test_size_percent=0.25, test_set_name='test')

    data_train = {
        "x": data["train"]["x"],
        "y": data["train"]["y"]
    }

    data_test = {
        "x": data["test"]["x"],
        "y": data["test"]["y"]
    }

    # scale data
    scaler = StandardScaler()
    scaler.fit(X)
    data_train["x"] = scaler.transform(data_train["x"])
    data_test["x"] = scaler.transform(data_test["x"])

    numberOfClasses = len(np.unique(y, axis=0))
    class_names = list(np.unique(y, axis=0))
    numberOfFeatures = len(data_train["x"][0])
    print("There are %i classes:" % numberOfClasses, class_names)
    print("There are %i features." % numberOfFeatures)

    nn = Network(MnistInputLayer())
    # nn.add(FullyConnectedLayer(100))
    # nn.add(Sigmoid())
    nn.add(FullyConnectedLayer(75))
    nn.add(Sigmoid())
    nn.add(FullyConnectedLayer(numberOfClasses))
    nn.add(Softmax())
    nn.add(CategoricalCrossEntropy())

    # nn.compile()

    # softmax = Softmax()
    # d = Tensor(elements=np.array([[2, 3, 0.2], [0.5, 2, 0.8]]))
    #
    # e = []
    # label = Tensor(elements=np.array([[1, 0, 0], [0, 1, 0]]))
    # softmax.forward([d], e)
    # actual_deltas = e[0].elements - label.elements
    # loss = CategoricalCrossEntropy()
    # loss.forward(e, [label])
    #
    # loss.backward([label], e)
    # softmax.backward(e, [d])

    trainer = SGDTrainer()
    trainer.optimize(network=nn, x=data_train["x"], y=data_train["y"], epochs=20, batch_size=1,
                     alpha=0.03,
                     validation_split=0.15, metrics=[Accuracy()])
