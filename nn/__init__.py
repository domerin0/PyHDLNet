from tensor import HDLTensor, HDLScalar
from typing import Any, List
from abc import ABC, abstractmethod


class NeuralModule(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.mode = 'train'
        self.sim = False
        self.requires_grad = False

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


class NeuralContainer(NeuralModule):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.modules = []
        self.input_dic = {}

    # @abstractmethod
    # def


class ReLU(NeuralModule):
    def __init__(self) -> None:
        super().__init__()
        self.zero_point = None

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        if self.zero_point is None:
            self.zero_point = 0 if inputs.signed else 2**inputs.bits // 2
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
        return "relu"

    def eval(self) -> None:
        super().eval()

    def train(self) -> None:
        super().train()


class SoftMax(NeuralModule):
    def __init__(self) -> None:
        super().__init__()
        self.one = HDLScalar(1)
        self.max_int = None
        self.acc = None

    def forward(self, inputs: HDLTensor) -> HDLTensor:
        if self.mode == 'eval':
            return inputs.argmax(axis=1)
        elif self.mode == 'train':
            if self.acc is None:
                self.max_int = HDLScalar(
                    2**inputs.bits if not inputs.signed else 2**(inputs.bits-1))
                self.acc = HDLTensor(
                    inputs.rows, inputs.columns, bits=inputs.bits+20
                )

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

    def backward(self, d_out: HDLTensor, x_in: HDLTensor) -> HDLTensor:
        raise NotImplementedError("SoftMax backward not implemented")

    def __repr__(self) -> str:
        return "SoftMax()"

    def __str__(self) -> str:
        return "softmax"

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

        self.grad_b = None
        self.grad_w = None
        self.requires_grad = False

    def forward(self, other: HDLTensor) -> HDLTensor:
        other @= self.weights.transpose()
        for i in range(other.rows):
            other.value[i, :] += self.bias.value
        if self.truncate_bits > 0:
            other.truncate_lsb(msb=self.truncate_bits,
                               acc_bits=2 * self.bits)
        return other

    def backward(self, d_out: HDLTensor, x_in: HDLTensor) -> HDLTensor:
        """
        Calculates dw, db for the linear layer
        Args:
            d_out :  Gradient of the cost with respect to the linear output. 
                     or the accumulated gradients from the prev layers. 
                     This is used for the chain rule to compute the gradients.
            x_in :   Original input to the layer. 
        Returns:
            dx : Gradient of cost wrt to the activation of the previous layer or the input of the 
                 current layer.
        """
        if self.grad_w is None:
            self.m = HDLScalar(x_in.columns)
        # gradient of loss wrt to the weights
        self.grad_w = x_in.transpose() @ d_out
        self.grad_w = self.grad_w // self.m
        # gradient of the loss wrt to the bias
        self.grad_b = d_out.sum(axis=0)
        self.grad_b = self.grad_b // self.m
        # gradient of the loss wrt to the input of the linear layer
        # this is used to continue the chain rule
        dx = d_out @ self.weights
        if self.truncate_bits > 0:
            self.grad_w.truncate_lsb(
                msb=self.truncate_bits,
            )
            self.grad_b.truncate_lsb(
                msb=self.truncate_bits,
            )
            dx.truncate_lsb(
                msb=self.truncate_bits,
            )

        return dx

    def update(self, lr_top: int, lr_bottom: int) -> None:
        self.weights -= lr_top * self.grad_w // lr_bottom
        self.bias -= lr_top * self.grad_b // lr_bottom

    def __repr__(self) -> str:
        return "Linear({0}, {1})".format(self.in_d, self.out_d)

    def __str__(self) -> str:
        return "linear_{0}_{1}".format(self.in_d, self.out_d)

    def eval(self) -> None:
        return super().eval()

    def zero_grad(self) -> None:
        self.grad_w.zero()
        self.grad_b.zero()

    def train(self) -> None:
        self.requires_grad = True
        return super().train()


class Sequential(NeuralModule):
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers_count = {}
        self.layers = [{
            "name": self.get_layer_name(l),
            "value": l
        }for l in list(layers)]

        self.setup_done = None

    def get_layer_name(self, layer: NeuralModule) -> str:
        if layer not in self.layers_count:
            self.layers_count[str(layer)] = 0
        self.layers_count[layer] += 1
        return "{0}_{1}".format(str(layer), self.layers_count[layer])

    def add(self, layer: NeuralModule) -> None:
        l = {
            "name": self.get_layer_name(layer),
            "value": layer
        }
        if self.mode == 'train':
            l.train()
        else:
            l.eval()

        self.layers.append(l)

    def forward(self, input: HDLTensor) -> HDLTensor:
        vec = input
        for layer in self.layers:
            if self.mode == 'train':
                layer["input"] = vec.copy()
            vec = layer["value"].forward(vec)
        return vec

    def backward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        vec = y_pred
        for layer in reversed(self.layers):
            grad_out = layer["value"].backward(layer["input"], vec)
        return vec

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        y = self.forward(*args, **kwds)
        if self.mode == 'train':
            self.backward(y, *args, **kwds)
        return y

    def __repr__(self) -> str:
        return "(\n{0}\n)".format(",\n".join([repr(l) for l in self.layers]))

    def __str__(self) -> str:
        return "sequential"

    def eval(self) -> None:
        super().eval()
        for layer in self.layers:
            layer.eval()

    def train(self) -> None:
        super().train()
        for layer in self.layers:
            layer.train()
