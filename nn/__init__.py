from tensor import HDLTensor
from typing import Any, List
from abc import ABC, abstractmethod


class NeuralModule(ABC):

    @abstractmethod
    def __init__(self) -> None:
        self.simulate = False

    @abstractmethod
    def forward(self, input: HDLTensor) -> HDLTensor:
        pass

    def __call__(self, input: HDLTensor) -> HDLTensor:
        return self.forward(input)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    def simulate(self) -> None:
        self.simulate = True


class ReLU(NeuralModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        for i in range(inputs.rows):
            for j in range(inputs.columns):
                inputs.ternary(
                    i, j, lambda x: x < 0, 0, inputs[i, j])
        return inputs

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def __repr__(self) -> str:
        return "ReLU()"

    def __str__(self) -> str:
        return "ReLU()"


class SoftMax(NeuralModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        sums = inputs.sum(axis=0)
        for row in range(sums.rows):
            inputs[row, 0] = inputs[row, 0] / sums[row]
        return inputs

    def __repr__(self) -> str:
        return "SoftMax()"

    def __str__(self) -> str:
        return "SoftMax()"

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class Linear(NeuralModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 precision_bits: int = 4,
                 weights: List[List[float]] = None,
                 bias: List[float] = None) -> None:
        super().__init__()
        if weights is None:
            self.weights = HDLTensor(
                input_size, output_size,
                bits=precision_bits
            )
        else:
            self.weights = HDLTensor(
                input_size, output_size,
                bits=precision_bits, value=weights
            )

        if bias is None:
            self.bias = HDLTensor(
                1, output_size, bits=precision_bits
            )
        else:
            self.bias = HDLTensor(
                1, output_size,
                bits=precision_bits, value=bias
            )
        self.input_size = input_size
        self.output_size = output_size

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def forward(self, other: HDLTensor) -> HDLTensor:
        other @= self.weights
        other += self.bias
        return other

    def __repr__(self) -> str:
        return "Linear({0}, {1})".format(self.input_size, self.output_size)

    def __str__(self) -> str:
        return "Linear({0}, {1})".format(self.input_size, self.output_size)


class Sequential(NeuralModule):
    def __init__(self, layers: List[NeuralModule]) -> None:
        super().__init__()
        self.layers = layers

    def add(self, layer: NeuralModule) -> None:
        self.layers.append(layer)

    def forward(self, input: HDLTensor) -> HDLTensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return "(\n{0}\n)".format(",\n".join([repr(l) for l in self.layers]))

    def __str__(self) -> str:
        return "(\n\t{0}\n)".format(",\n\t".join([str(l) for l in self.layers]))

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass
