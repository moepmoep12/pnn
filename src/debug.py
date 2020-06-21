from Trainer import *
from Layer.Layer import *
from NeuralNetwork import *
from Layer.ActivationLayer import *
from Layer.Loss import *

if __name__ == '__main__':
    import numpy as np


class InLayer(InputLayer):

    def forward(self, raw_data):
        return [Tensor(elements=np.array([[0.4183, 0.5209, 0.0291]]))], [Tensor(elements=np.array([0.7095, 0.0942]))]


nn = Network(InLayer())
nn.layers.append(FullyConnectedLayer(3))
nn.layers[-1].biases.elements = np.zeros(nn.layers[-1].biases.elements.shape)
nn.layers[-1].weights = Tensor(elements=np.array([
    [-0.5057, 0.3987, -0.8943],
    [0.3356, 0.1673, 0.8321],
    [-0.3485, -0.4597, -0.1121]]))

nn.layers.append(Sigmoid())

nn.layers.append(FullyConnectedLayer(2))
nn.layers[-1].biases.elements = np.zeros(nn.layers[-1].biases.elements.shape)
nn.layers[-1].weights = Tensor(elements=np.array([
    [0.4047, 0.9563],
    [-0.8192, -0.1274],
    [0.3662, -0.7252]]))

nn.layers.append(Softmax())
nn.layers.append(CategoricalCrossEntropy())

tensors = nn.forward(None)
deltas = nn.backprop(tensors)
deltas.reverse()
