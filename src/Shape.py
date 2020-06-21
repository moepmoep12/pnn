from typing import List
import numpy as np


class Shape:
    def __init__(self, axis: List[int]):
        """
        :param axis: array of ints specifying the maximum length of each dimension in column-major representation
                    e.g. [1,3] refers to a shape with 1 row & 3 columns like [0 1 2]
        """
        self.axis = np.array(axis)

    def size(self) -> int:
        """
        :return: the maximum number of elements
        """
        return np.prod(self.axis)
