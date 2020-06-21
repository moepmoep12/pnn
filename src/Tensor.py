import numpy as np
from Shape import Shape


class Tensor:
    """
    stores floats in the elements array with the given shape
    """

    def __init__(self, shape: Shape = None, elements: np.ndarray = None):
        """"
        :param shape: The shape of the tensor.
        :param init_random: Whether the elements will be initialized with random numbers.
        """
        self.deltas = None
        if shape is not None:
            self.shape = shape

        if elements is not None:
            self.elements = elements
            if shape is None:
                self.shape = Shape(elements.shape)
        else:
            self.elements = np.random.randn(*self.shape.axis) #np.empty(shape=self.shape.axis, dtype=np.float64)

    def get_deltas(self) -> np.dtype:
        """
        Lazy initialization of deltas.
        """
        if self.deltas is None:
            self.deltas = np.zeros(self.shape)
        else:
            return self.deltas

    def get_shape(self) -> Shape:
        return self.shape

    def __str__(self) -> str:
        return self.elements.__str__()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(elements=np.add(self.elements, other.elements))
        else:
            return Tensor(elements=np.add(self.elements, other))

    def __iadd__(self, other):
        if isinstance(other, Tensor):
            self.elements += other.elements
        else:
            self.elements += other
        return self

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(elements=np.subtract(self.elements, other.elements))
        else:
            return Tensor(elements=np.subtract(self.elements, other))

    def __isub__(self, other):
        if isinstance(other, Tensor):
            self.elements = np.subtract(self.elements, other.elements)
        else:
            self.elements -= other
        return self

    def __mul__(self, other):
        """
        element-wise multiplication
        """
        if isinstance(other, Tensor):
            return Tensor(elements=np.multiply(self.elements, other.elements))
        else:
            return Tensor(elements=np.multiply(self.elements, other))
