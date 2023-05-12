from __future__ import annotations
import pyrtl
from pyrtl.rtllib.matrix import Matrix
import pyrtl.rtllib.matrix as matrix
from pyrtl.corecircuits import select, shift_left_logical, signed_ge
from utility import quantize, uniform_int
import random
from abc import ABC, abstractmethod


class Tensor(ABC):
    def __init__(self):
        pass


class HDLTensor:
    def __init__(self, *dims, value=None, bits=8, signed=False) -> None:
        if len(dims) > 2:
            raise ValueError("HDLTensor only supports 2D matrices")

        if signed:
            self.max_int = 2 ** (bits - 1) - 1
            self.min_int = -(2 ** (bits - 1))
        else:
            self.max_int = 2 ** bits - 1
            self.min_int = 0

        if value is None:
            rand = uniform_int(self.max_int, self.min_int, dims[0], dims[1])
            self.value = Matrix(dims[0], dims[1], bits, value=rand)
        elif isinstance(value, pyrtl.Input) or isinstance(value, list):
            self.value = Matrix(dims[0], dims[1], bits, value=value)
        elif isinstance(value, str) and value == 'input':
            self.value = Matrix(
                dims[0],
                dims[1],
                bits,
                value=pyrtl.Input(
                    bitwidth=dims[0]*dims[1]*bits, name='net_input')
            )
        else:
            self.value = value

        self.signed = signed
        self.shape = dims

        # auto grad related properties
        self.requires_grad = False
        self.grad = None
        self.grad_fn = None
        self.is_leaf = True

    def argmax(self, axis: int) -> HDLTensor:
        mat = matrix.argmax(self.value, axis=axis)
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    @staticmethod
    def multiply(first: HDLTensor, second: HDLTensor) -> HDLTensor:
        mat = matrix.multiply(first.value, second.value)

        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def sum(self, axis: int) -> HDLTensor:
        mat = matrix.sum(self.value, axis=axis)
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def max(self, axis: int):
        val = matrix.max(self.value, axis=axis)
        return HDLTensor(val.rows, val.columns, bits=val.bits, value=val)

    def truncate_lsb(self, bits: int):
        '''
        Truncates the least significant bits of the tensor.
        Keeps only msb bits from the tensor.
        '''
        for i in range(self.rows):
            for j in range(self.columns):
                self.value[i, j] = self.value[i, j][:bits]
        self.value.bits = bits

    def divide_nbit_unsigned(bits, num, divisor):
        # Initialize quotient as 0
        quotient = 0
        # Initialize remainder as 0
        remainder = 0

        # Iterate through each bit
        for i in range(bits):
            # Left-shift remainder and add next bit of num
            remainder = (remainder << 1) | ((num >> (bits - 1 - i)) & 1)
            if remainder >= divisor:
                remainder -= divisor
                # Set corresponding quotient bit if division is possible
                quotient |= (1 << (bits - 1 - i))

        return quotient

    def __floordiv__(self, b: HDLTensor) -> HDLTensor:
        '''
        Perform the element-wise division op.
        b must match the dimensions of self or be a scalar (1x1 tensor). 
        '''
        if not isinstance(b, HDLTensor):
            raise ValueError("Cannot divide HDLTensor by non-HDLTensor")
        if b.rows != self.rows or b.columns != self.columns \
                and b.rows != 1 and b.columns != 1:
            raise ValueError("Cannot divide HDLTensor of different dimensions")
        quotients = self.value

        for i in range(self.rows):
            for j in range(self.columns):
                divisor = b[0, 0] if b.rows == 1 and b.columns == 1 else b[i, j]
                quotients[i, j] = quotients[i, j] // divisor
                # ge = select(quotients[i, j] >= divisor, True, False)
                # print(ge)
                # while ge:
                #     quotients[i, j] = quotients[i,j] - divisor

        return HDLTensor(quotients.rows, quotients.columns, bits=quotients.bits, value=quotients)

    def __truediv__(self, b: HDLTensor) -> HDLTensor:
        '''
        Perform the element-wise division op.
        b must match the dimensions of self or be a scalar (1x1 tensor). 
        '''
        return self.__floordiv__(b)

    def reshape(self, *newshape, **order) -> HDLTensor:
        return self.value.reshape(self.value, *newshape, **order)

    def ternary(self, row: int, col: int, cond_fn: function, true, false) -> None:
        self.value[row, col] = select(
            cond_fn(self.value[row, col]), true, false
        )

    @property
    def bits(self):
        return self.value.bits

    @property
    def columns(self):
        return self.value.columns

    @property
    def rows(self):
        return self.value.rows

    def to_wirevector(self):
        return self.value.to_wirevector()

    def __matmul__(self, other: HDLTensor) -> HDLTensor:
        mat = self.value.__matmul__(other.value)
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __mul__(self, other: HDLTensor):
        if isinstance(other, float) or isinstance(other, int):
            mat = self.value * other.value
        mat = self.value * other.value
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def __imul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.value *= other
        self.value *= other.value
        return self

    def __add__(self, other: HDLTensor) -> HDLTensor:
        return HDLTensor(
            self.rows, self.columns,
            bits=self.bits,
            value=self.value + other.value
        )

    def exp(self):
        ''' 
        Perform the element-wise exp op.

        :return: a Matrix object with the element wise exp being performed
        '''
        if not isinstance(self.value, Matrix):
            return 2 ** self.value
        result = Matrix(self.rows, self.columns,
                        self.value.bits + 1, max_bits=self.value.max_bits)

        for i in range(result.rows):
            for j in range(result.columns):
                result[i, j] = shift_left_logical(self[i, j], 1)
        return result

    def __iadd__(self, other):
        self.value += other.value
        return self

    def __sub__(self, other):
        mat = self.value - other.value
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def __isub__(self, other):
        self.value -= other.value
        return self

    def __pow__(self, other: int):
        return self.value ** other

    def __ipow__(self, other):
        self.value **= other
        return self

    def __getitem__(self, tup):
        x, y = tup
        if isinstance(self.value, Matrix):
            return self.value[x, y]
        raise ValueError("HDLTensor value is not a Matrix")

    def __setitem__(self, tup, value):
        x, y = tup
        self.value[x, y] = value

    def transpose(self):
        val = self.value.transpose()
        return HDLTensor(
            val.rows,
            val.columns,
            bits=val.bits,
            value=val
        )


class PyTensor:
    def __init__(self,
                 *dims, value=None,
                 bits=8, signed=False,
                 quantize_val=True,
                 zero_points=None,
                 scale_factors=None) -> None:
        '''
        This is just a data holding class we can import into 
        pure python code as it doesn't rely on the PyRTL library.
        '''
        if len(dims) > 2:
            raise ValueError("PyTensor only supports 2D matrices")
        if quantize_val:
            if zero_points is None:
                raise ValueError("PyTensor requires zero_points")
            if scale_factors is None:
                raise ValueError("PyTensor requires scale_factors")
            if len(zero_points) != len(scale_factors) or len(zero_points) != dims[-1]:
                raise ValueError(
                    "PyTensor requires {0} number of zero_points ({1}) and scale_factors ({2})"
                    .format(len(dims[-1]), len(zero_points), len(scale_factors))
                )

        if signed:
            self.max_int = 2 ** (bits - 1) - 1
            self.min_int = -2 ** (bits - 1)
        else:
            self.max_int = 2 ** bits - 1
            self.min_int = 0

        if value is None:
            rand = uniform_int(self.max_int, self.min_int, dims[0], dims[1])
            self.value = rand
        else:
            self.value = value

        if quantize_val:
            for i in range(dims[0]):
                for j in range(dims[1]):
                    self.value[i][j] = quantize(
                        value[i][j],
                        scale_factors[j],
                        zero_points[j],
                        self.max_int, self.min_int
                    )

        self.signed = signed
        self.bits = bits
        self.rows = dims[0]
        self.columns = dims[1]
