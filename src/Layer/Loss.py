from Layer.Layer import Layer
from Tensor import Tensor
from Shape import Shape
import numpy as np
from typing import List


class CategoricalCrossEntropy(Layer):

    def __init__(self):
        self.labels = []
        self.out_shape = Shape([1])

    @staticmethod
    def categorical_cross_entropy(h_one_hot, y_one_hot):
        h_one_hot = np.clip(h_one_hot, a_min=0.000000001, a_max=None)
        return np.mean(-np.sum(y_one_hot * np.log(h_one_hot), axis=1))

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        self.labels = []
        for i in range(len(in_tensors)):
            if i >= len(out_tensors):
                out_tensors.append(Tensor(self.out_shape))
            self.labels.append(out_tensors[i].elements)
            out_tensors[i].elements = self.categorical_cross_entropy(in_tensors[i].elements, self.labels[i])

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            h_one_hot = np.clip(in_tensors[i].elements, a_min=0.000000001, a_max=None)
            in_tensors[i].deltas = -self.labels[i] / h_one_hot


class CategoricalCrossEntropyWithSoftMax(Layer):

    def __init__(self):
        self.labels = []
        self.out_shape = Shape([1])

    @staticmethod
    def categorical_cross_entropy(h_one_hot, y_one_hot):
        h_one_hot = np.clip(h_one_hot, a_min=0.000000001, a_max=None)
        return np.mean(-np.sum(y_one_hot * np.log(h_one_hot), axis=1))

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        self.labels = []
        for i in range(len(in_tensors)):
            if i >= len(out_tensors):
                out_tensors.append(Tensor(self.out_shape))
            self.labels.append(out_tensors[i].elements)
            out_tensors[i].elements = self.categorical_cross_entropy(in_tensors[i].elements, self.labels[i])

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        """
        When using cross entropy with softmax the deltas simplify to ( prediction - ground truth )
        """
        for i in range(len(in_tensors)):
            in_tensors[i].deltas = in_tensors[i].elements - self.labels[i]


class CrossEntropy(Layer):

    def __init__(self):
        self.labels = []
        self.out_shape = Shape([1])

    @staticmethod
    def cross_entropy(h, y):
        h = np.clip(h, 0.000000001, 0.99999999)
        return np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))

    def get_input_shape(self) -> Shape:
        return self.in_shape

    def get_output_shape(self) -> Shape:
        return self.out_shape

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        self.labels = []
        for i in range(len(in_tensors)):
            if i >= len(out_tensors):
                out_tensors.append(Tensor(self.out_shape))
            self.labels.append(out_tensors[i].elements)
            out_tensors[i].elements = self.cross_entropy(in_tensors[i].elements, out_tensors[i].elements)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            h = np.clip(in_tensors[i].elements, 0.000000001, 0.99999999)
            in_tensors[i].deltas = ((1 - self.labels[i]) / (1 - h)) - self.labels[i] / h


class MeanSquaredError(Layer):

    def __init__(self):
        self.labels = []

    def set_input_shape(self, input_shape: Shape):
        self.in_shape = Shape(input_shape.axis)
        self.out_shape = Shape([1])

    def mse(self, h, y):
        return np.mean(np.square(h - y))

    def get_input_shape(self) -> Shape:
        return self.in_shape

    def get_output_shape(self) -> Shape:
        return self.out_shape

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            out_tensors[i].elements = self.mse(in_tensors[i].elements, self.y)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # todo
        pass
