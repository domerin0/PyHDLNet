from tensor import HDLTensor, HDLScalar
from typing import Any, List
from abc import ABC, abstractmethod
import pyrtl.rtllib.matrix as matrix


class NeuralModule(ABC):

    @abstractmethod
    def __init__(self) -> None:
        self.mode = 'train'
        self.sim = False

    @abstractmethod
    def forward(self, x: HDLTensor) -> HDLTensor:
        pass

    @abstractmethod
    def backward(self, y_pred: HDLTensor, y: HDLTensor) -> HDLTensor:
        pass

    def __call__(self, x: HDLTensor) -> HDLTensor:
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

    def backward(self, d_out: HDLTensor, x_in: HDLTensor) -> HDLTensor:
        for i in range(d_out.rows):
            for j in range(d_out.columns):
                d_out.ternary(
                    i, j,
                    lambda _: x_in[i, j] < self.zero_point,
                    self.zero_point,
                    d_out[i, j]
                )
        return d_out

    def __repr__(self) -> str:
        return "ReLU()"

    def __str__(self) -> str:
        return "ReLU()"

    def eval(self) -> None:
        super().eval()

    def train(self) -> None:
        super().train()


class SoftMax(NeuralModule):
    def __init__(self, output_dim=None, batch_size=1, signed=False, bits=8) -> None:
        super().__init__()
        self.one = HDLScalar(1)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_int = HDLScalar(2**bits if not signed else 2**(bits-1))
        self.acc = None
        if output_dim is not None:
            self.acc = HDLTensor(batch_size, output_dim, bits=bits+20)

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        if self.mode == 'eval':
            return inputs.argmax(axis=1)
        elif self.mode == 'train':
            if self.acc is None:
                raise ValueError(
                    "Must provide output_dim to SoftMax for training")

            # second order taylor approximation
            for i in range(inputs.rows):
                for j in range(inputs.columns):
                    self.acc[i, j] = inputs[i, j] * inputs[i, j]
                    self.acc.shift_elem_right(1, i, j)
                    self.acc[i, j] = self.acc[i, j] + 1 + inputs[i, j]

            # sum up rows
            sums = self.acc.sum(axis=1)

            # scale up by max_int, divide by sum
            for i in range(self.acc.rows):
                for j in range(self.acc.columns):
                    self.acc[i, j] = self.acc[i, j] * self.max_int.value
                    self.acc[i, j] = self.acc.divide_scalar(
                        divisor=sums[0, i],
                        row=i,
                        col=j
                    )

            # return estimated softmax
            return self.acc

        raise ValueError(
            "Mode {0} not supported, use train() or eval()".format(self.mode))

    def backward(self, y_pred: HDLTensor, y: HDLTensor) -> HDLTensor:
        raise NotImplementedError("SoftMax backward not implemented")

    def __repr__(self) -> str:
        return "SoftMax()"

    def __str__(self) -> str:
        return "SoftMax()"

    def eval(self) -> None:
        super().eval()

    def train(self) -> None:
        super().train()


class Linear(NeuralModule):
    def __init__(self,
                 in_d: int,
                 out_d: int,
                 bits: int = 8,
                 truncate_bits: int = 0,
                 signed=False,
                 weights: List[List[float]] = None,
                 bias: List[float] = None) -> None:
        super().__init__()

        if truncate_bits >= bits**2:
            raise ValueError(
                "Truncate bits {0} cannot be greater than precision bits {1}"
                .format(truncate_bits, bits)
            )

        if weights is None:
            self.weights = HDLTensor(
                out_d,
                in_d,
                bits=bits,
                signed=signed
            )
        else:
            self.weights = HDLTensor(
                out_d,
                in_d,
                bits=bits,
                value=weights,
                signed=signed
            )

        if bias is None:
            self.bias = HDLTensor(
                1,
                out_d,
                value=[[0 for _ in range(out_d)]],
                bits=bits,
                signed=signed
            )
        else:
            self.bias = HDLTensor(
                1,
                out_d,
                bits=bits,
                value=bias,
                signed=signed
            )

        self.bits = bits
        self.truncate_bits = truncate_bits
        self.in_d = in_d
        self.out_d = out_d

    def forward(self, other: HDLTensor) -> HDLTensor:
        other @= self.weights.transpose()
        for i in range(other.rows):
            other.value[i, :] += self.bias.value
        if self.truncate_bits > 0:
            other.truncate_lsb(msb=self.truncate_bits,
                               acc_bits=2 * self.bits)
        return other

    def backward(self, dout: HDLTensor, x_in: HDLTensor) -> HDLTensor:
        if self.backprop_ones is None:
            self.backprop_ones = HDLTensor(
                x_in.rows,
                dout.rows,
                bits=self.bits,
                value=[[1 for i in range(dout.rows)] for _ in range(x_in.rows)]
            )
        db += self.backprop_ones @ dout

        dw += x_in.transpose() @ dout
        dx = x_in.transpose() @ self.weights
        return dx

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

    def backward(self, y_pred: HDLTensor, y: HDLTensor) -> HDLTensor:
        vec = input
        for layer in reversed(self.layers):
            vec = layer.backward(vec)
        return vec

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        y = self.forward(*args, **kwds)
        if self.mode == 'train':
            self.backward(y, *args, **kwds)
        return y

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
