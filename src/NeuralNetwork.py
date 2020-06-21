from Layer.Layer import InputLayer
from Tensor import Tensor
from typing import List


class Network:
    def __init__(self, input_layer: InputLayer):
        self.layers = []
        self.input_layer = input_layer

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, raw_data) -> List[Tensor]:
        in_tensors, label = self.input_layer.forward(raw_data)
        tensors = [in_tensors]
        for i in range(len(self.layers)):
            # loss
            if i == len(self.layers) - 1:
                out_tensors = label
            else:
                out_tensors = []

            self.layers[i].forward(in_tensors, out_tensors)
            tensors.append(out_tensors)
            in_tensors = out_tensors

        return tensors

    def backprop(self, tensors) -> List[List[Tensor]]:
        deltas = []
        for i in reversed(range(len(self.layers))):
            self.layers[i].backward(tensors[i + 1], tensors[i])
            deltas.append(self.layers[i].calculate_delta_weights(tensors[i + 1], tensors[i]))
        return deltas

    def predict(self, raw_data) -> Tensor:
        tensor = self.forward(raw_data)[-2][0]
        tensor.elements = tensor.elements.round()
        return tensor

    # def compile(self):
    #     out_shape = self.input_layer.get_output_shape()
    #     for layer in self.layers:
    #         layer.set_input_shape(out_shape)
    #         out_shape = layer.get_output_shape()
