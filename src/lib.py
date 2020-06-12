import numpy as np
from enum import Enum
from typing import List


class Shape:
    def __init__(self, axis: List[int]):
        """
        :param axis: array of ints specifying the maximum length of each dimension in column-major representation
                    e.g. [1,3] refers to a shape with 1 row & 3 columns like [0 1 2]
        """
        self.axis = axis

    def size(self) -> int:
        """
        :return: the maximum number of elements
        """
        return np.prod(self.axis)


class Tensor:
    """
    stores floats in the elements array with the given shape
    """

    def __init__(self, shape: Shape, init_random=False):
        """"
        :param shape: The shape of the tensor.
        :param init_random: Whether the elements will be initialized with random numbers.
        """
        self.deltas = None
        if init_random:
            self.elements = np.random.randn(self.shape.axis)
        else:
            self.elements = np.empty(shape=self.shape, dtype=np.float64)
        pass

    def get_deltas(self) -> np.dtype:
        """
        Lazy initialization of deltas.
        """
        if self.deltas is None:
            self.deltas = np.zeros(self.shape)
        else:
            return self.deltas


class InputLayer:
    def forward(self, raw_data) -> List[Tensor]:
        """
        Transforms the raw_data into tensors.
        :param raw_data:
        :return: a list of tensors.
        """
        # result = []
        # tensor = Tensor(Shape(raw_data.shape()))
        # tensor.elements = raw_data
        # result.append(tensor)
        # return result
        pass


class Layer:
    """
    Layer interface.
    """

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        """
        Use elements of in_tensors to calculate elements in out_tensors.
        :param in_tensors: List of input tensors.
        :param out_tensors: The tensors after going through this layer.
        """
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        """
        Use deltas of out_tensors to calculate deltas of in_tensors.
        :param in_tensors: List of incoming tensors.
        :param out_tensors: List of outgoing tensors.
        """
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        """
        Use elements of in_tensors and deltas of out_tensors to calculate delta_weights
        :param out_tensors: a list of incoming tensors
        :param in_tensors: a list of outgoing tensors
        """
        pass


class FullyConnectedLayer(Layer):
    """
    A fully connected layer.
    """

    def __init__(self, nb_neurons: int, in_shape: Shape):
        self.in_shape = in_shape
        self.out_shape = Shape([nb_neurons])
        self.weights = Tensor(Shape([nb_neurons, in_shape.size()]), init_random=True)
        self.biases = Tensor(Shape([nb_neurons, 1]), init_random=True)

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            out_tensors[i] = in_tensors[i].elements.dot(self.weights) + self.biases

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class Padding(Enum):
    """
    Padding used in Convolutional Layer.
    """
    NONE = 0
    HALF = 1
    FULL = 2


class Conv2DLayer(Layer):
    """
    A convolutional layer.
    """
    def __init__(self, kernel_tensor: Tensor, padding: Padding):
        """
        To-Do: Extend Signature according to slides.
        :param kernel_tensor: 4-dim tensor that forms the weights of this layer
        """
        self._kernel_tensor = kernel_tensor
        self._padding = padding

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        """
        Y = InputTensor * KernelTensor + Bias, where '*' is the convolution operator
        :param in_tensors:
        :param out_tensors:
        :return:
        """
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class FlattenLayer(Layer):
    """
    A utility layer for flattening the tensors into linear shape.
    """
    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        # copies the values into outTensor ( which is in linear shape )
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # copies the deltas
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # empty
        pass


class CloneLayer(Layer):
    """
    A utility layer for cloning tensors.
    """
    def __init__(self, amount_clones: int):
        self._amount_clones = amount_clones

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        # input is single tensor and output is list of clones of that tensor
        for i in range(0, self._amount_clones):
            # clone
            pass

        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # sums the deltas at the respective positions
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # empty
        pass


class FilterLayer(Layer):
    """
    A utility layer for filtering input tensors.
    """
    def __init__(self, accepted_tensor_indices: List[int]):
        self._accepted_tensor_indices = accepted_tensor_indices
        pass

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        # input is multiple tensors and output is a subset of the incoming tensors
        for i in range(0, len(in_tensors)):
            if i in self._accepted_tensor_indices:
                out_tensors[i] = in_tensors[i]

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # maps the deltas to the incoming tensors
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class ConcatLayer(Layer):
    """
    A utility layer for concatenating tensors.
    """
    pass


class TransposeLayer(Layer):
    """
    A utility layer for transposing tensors.
    """
    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class ActivationLayer(Layer):
    """
    Abstract ActivationLayer
    """

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class Sigmoid(ActivationLayer):

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            out_tensors[i] = np.clip(in_tensors[i].elements, -500, 500)
            out_tensors[i] = np.exp(out_tensors[i]) / (1 + np.exp(out_tensors[i]))

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class Network:
    def __init__(self, input_layer: InputLayer):
        self.layers = []
        self.input_layer = input_layer

    def forward(self, raw_data):
        in_tensors = self.input_layer.forward(raw_data)
        out_tensors = []
        for i in len(self.layers):
            self.layers[i].forward(in_tensors, out_tensors)

    def backprop(self):
        pass

    pass


class SGDTrainer:
    # batchSize
    # learningRate
    # epochs
    # shuffle
    # sgdFlavor
    def optimize(self, network: Network, data):
        pass
