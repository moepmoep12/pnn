from Tensor import Tensor
from Shape import Shape
from typing import List
from enum import Enum
import numpy as np


class Layer:
    """
    Layer interface.
    """

    # def get_input_shape(self) -> Shape:
    #     """
    #     Returns the shape of the input to this layer.
    #     """
    #     pass
    #
    # def set_input_shape(self, input_shape: Shape):
    #     pass
    #
    # def get_output_shape(self) -> Shape:
    #     """
    #     Returns the shape of the output of this layer.
    #     """
    #     pass

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

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]) -> List[Tensor]:
        """
        Use elements of in_tensors and deltas of out_tensors to calculate delta_weights
        :param out_tensors: a list of incoming tensors
        :param in_tensors: a list of outgoing tensors
        """
        return None

    def update_parameter(self, parameter: List[Tensor]):
        pass


class InputLayer:
    """
    InputLayer Interface.
    """

    # def get_output_shape(self) -> Shape:
    #     pass

    def forward(self, raw_data):
        pass


class FullyConnectedLayer(Layer):
    """
    A fully connected layer.
    """

    def __init__(self, nb_neurons: int):
        self.nb_neurons = nb_neurons
        self.biases = Tensor(shape=Shape([1, self.nb_neurons]))
        self.weights = None
        self.out_shape = None

    # def set_input_shape(self, input_shape: Shape):
    #     self.in_shape = input_shape
    #     self.weights = Tensor(Shape([self.nb_neurons, input_shape.size()]))
    #
    # def get_input_shape(self) -> Shape:
    #     return self.in_shape
    #
    # def get_output_shape(self) -> Shape:
    #     return self.out_shape

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        if self.out_shape is None:
            self.out_shape = Shape([in_tensors[0].get_shape().axis[0], self.nb_neurons])
        if self.weights is None:
            self.weights = Tensor(shape=Shape([in_tensors[0].get_shape().axis[1], self.nb_neurons]))

        for i in range(len(in_tensors)):
            if i >= len(out_tensors):
                out_tensors.append(Tensor(self.out_shape))

            out_tensors[i].elements = in_tensors[i].elements.dot(self.weights.elements) + self.biases.elements

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            in_tensors[i].deltas = np.dot(out_tensors[i].deltas, self.weights.elements.transpose())

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]) -> List[Tensor]:
        batch_size = in_tensors[0].elements.shape[0]
        dw = np.dot(in_tensors[0].elements.transpose(), out_tensors[0].deltas) / float(batch_size)
        db = np.dot(np.ones((1, batch_size)), out_tensors[0].deltas) / float(batch_size)
        return [Tensor(elements=dw), Tensor(elements=db)]

    def update_parameter(self, parameter: List[Tensor]):
        self.weights -= parameter[0]
        self.biases -= parameter[1]


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
        # todo
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # todo
        pass
