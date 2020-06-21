from Layer.Layer import Layer
from typing import List
from Tensor import Tensor
import numpy as np


class ActivationLayer(Layer):
    """
    Abstract ActivationLayer
    """

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass


class Sigmoid(ActivationLayer):

    @staticmethod
    def sigmoid(z):
        val = np.clip(z, -500, 500)  # Prevent overflow
        val = np.exp(val) / (1 + np.exp(val))
        return val

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            if i >= len(out_tensors):
                out_tensors.append(Tensor(shape=in_tensors[i].get_shape()))
            out_tensors[i].elements = self.sigmoid(in_tensors[i].elements)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for i in range(len(out_tensors)):
            in_tensors[i].deltas = np.multiply(np.multiply(out_tensors[i].elements, (1 - out_tensors[i].elements)),
                                               out_tensors[i].deltas)


class Softmax(ActivationLayer):

    @staticmethod
    def softmax(z):
        exps = np.exp(z - np.max(z))
        return exps / exps.sum(axis=1, keepdims=True)

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for i in range(len(in_tensors)):
            if i >= len(out_tensors):
                out_tensors.append(Tensor(in_tensors[i].get_shape()))
            out_tensors[i].elements = self.softmax(in_tensors[i].elements)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for i in range(len(out_tensors)):
            # off-diagonal Jacobian for every sample
            J = - out_tensors[i].elements[..., None] * out_tensors[i].elements[:, None, :]
            iy, ix = np.diag_indices_from(J[0])
            # diagonal entries
            J[:, iy, ix] = out_tensors[i].elements * (1.0 - out_tensors[i].elements)

            # calculate the error for every sample
            in_tensors[i].deltas = np.empty(out_tensors[0].elements.shape)
            for j in range(J.shape[0]):
                in_tensors[i].deltas[j, :] = np.dot(out_tensors[i].deltas[j, :], J[j])

    class SoftmaxWithCrossEntropy(ActivationLayer):

        @staticmethod
        def softmax(z):
            exps = np.exp(z - np.max(z))
            return exps / exps.sum(axis=1, keepdims=True)

        def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
            for i in range(len(in_tensors)):
                if i >= len(out_tensors):
                    out_tensors.append(Tensor(in_tensors[i].get_shape()))
                out_tensors[i].elements = self.softmax(in_tensors[i].elements)

        def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
            """
            When using softmax with cross entropy as loss, use the deltas from the loss, that is prediction - ground_truth
            """
            for i in range(len(out_tensors)):
                in_tensors[i].deltas = out_tensors[i].deltas
