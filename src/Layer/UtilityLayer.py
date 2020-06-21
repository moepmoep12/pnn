from Layer import Layer

from Tensor import Tensor
from Shape import Shape
from typing import List


class CloneLayer(Layer):
    """
    A utility layer for cloning tensors.
    """

    def __init__(self, amount_clones: int):
        self._amount_clones = amount_clones

    def set_input_shape(self, input_shape: Shape):
        self.in_shape = Shape(input_shape.axis)
        self.out_shape = Shape(input_shape.axis)

    def get_input_shape(self) -> Shape:
        return self.in_shape

    def get_output_shape(self) -> Shape:
        return self.out_shape

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


class FilterLayer(Layer):
    """
    A utility layer for filtering input tensors.
    """

    def __init__(self, accepted_tensor_indices: List[int]):
        self._accepted_tensor_indices = accepted_tensor_indices

    def set_input_shape(self, input_shape: Shape):
        self.in_shape = Shape(input_shape.axis)
        self.out_shape = Shape(input_shape.axis)

    def get_input_shape(self) -> Shape:
        return self.in_shape

    def get_output_shape(self) -> Shape:
        return self.out_shape

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
