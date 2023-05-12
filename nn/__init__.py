from tensor import HDLTensor, PyTensor
from typing import Any, List
from abc import ABC, abstractmethod
import pyrtl.rtllib.matrix as matrix


class NeuralModule(ABC):

    @abstractmethod
    def __init__(self) -> None:
        self.mode = 'train'
        self.sim = False

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
        self.mode = 'eval'

    @abstractmethod
    def train(self) -> None:
        self.mode = 'train'

    def simulate(self) -> None:
        self.sim = True


class ReLU(NeuralModule):
    def __init__(self, zero_point=0) -> None:
        super().__init__()
        self.zero_point = zero_point

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        for i in range(inputs.rows):
            for j in range(inputs.columns):
                inputs.ternary(
                    i, j,
                    lambda x: x < self.zero_point,
                    self.zero_point,
                    inputs[i, j]
                )
        return inputs

    def __repr__(self) -> str:
        return "ReLU()"

    def __str__(self) -> str:
        return "ReLU()"

    def eval(self) -> None:
        super().eval()

    def train(self) -> None:
        super().train()


class SoftMax(NeuralModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        if self.mode == 'eval':
            return inputs.argmax(axis=1)
        elif self.mode == 'train':
            raise NotImplementedError("SoftMax not implemented for training")

    def __repr__(self) -> str:
        return "SoftMax()"

    def __str__(self) -> str:
        return "SoftMax()"

    def eval(self) -> None:
        super().eval()

    def train(self) -> None:
        super().train()


class TruncateLSB(NeuralModule):
    def __init__(self, bits: int) -> None:
        super().__init__()
        self.bits = bits

    def forward(self, other: HDLTensor) -> HDLTensor:
        other.truncate_lsb(bits=self.bits)
        return other

    def __repr__(self) -> str:
        return "TruncateLSB(bits={0})".format(self.bits)

    def __str__(self) -> str:
        return "TruncateLSB(bits={0})".format(self.bits)

    def eval(self) -> None:
        super().eval()

    def train(self) -> None:
        return super().train()


class Linear(NeuralModule):
    def __init__(self,
                 in_d: int,
                 out_d: int,
                 precision_bits: int = 4,
                 signed=False,
                 weights: List[List[float]] = None,
                 bias: List[float] = None) -> None:
        super().__init__()
        if weights is None:
            self.weights = HDLTensor(
                out_d,
                in_d,
                bits=precision_bits,
                signed=signed
            )
        else:
            self.weights = HDLTensor(
                out_d,
                in_d,
                bits=precision_bits, value=weights,
                signed=signed
            )

        if bias is None:
            self.bias = HDLTensor(
                1,
                out_d,
                bits=precision_bits,
                signed=signed
            )
        else:
            self.bias = HDLTensor(
                1,
                out_d,
                bits=precision_bits,
                value=bias,
                signed=signed
            )

        self.in_d = in_d
        self.out_d = out_d

    def forward(self, other: HDLTensor) -> HDLTensor:
        other @= self.weights.transpose()
        other += self.bias
        return other

    def __repr__(self) -> str:
        return "Linear({0}, {1})".format(self.in_d, self.out_d)

    def __str__(self) -> str:
        return "Linear({0}, {1})".format(self.in_d, self.out_d)

    def eval(self) -> None:
        return super().eval()

    def train(self) -> None:
        return super().train()


class Sequential(NeuralModule):
    def __init__(self, layers: List[NeuralModule]) -> None:
        super().__init__()
        self.layers = layers
        self.setup_done = None

    def add(self, layer: NeuralModule) -> None:
        self.layers.append(layer)

    def forward(self, input: HDLTensor) -> HDLTensor:
        vec = input
        for layer in self.layers:
            vec = layer.forward(vec)
        return vec

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return "(\n{0}\n)".format(",\n".join([repr(l) for l in self.layers]))

    def __str__(self) -> str:
        return "(\n\t{0}\n)".format(",\n\t".join([str(l) for l in self.layers]))

    def eval(self) -> None:
        super().eval()
        for layer in self.layers:
            layer.eval()

    def train(self) -> None:
        super().train()
        for layer in self.layers:
            layer.train()
