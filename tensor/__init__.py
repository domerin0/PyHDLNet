from __future__ import annotations
import pyrtl
from pyrtl.rtllib.matrix import Matrix
import pyrtl.rtllib.matrix as matrix
from pyrtl.corecircuits import select, shift_left_logical

import random


class HDLTensor:
    def __init__(self, *dims, value=None, bits=8, signed=False) -> None:
        if len(dims) > 2:
            raise ValueError("HDLTensor only supports 2D matrices")

        if value is None:
            max_int = 2**bits - 1 if not signed else 2 ** (bits - 1) - 1
            rand = [[random.randint(0, max_int) for _ in range(dims[1])]
                    for _ in range(dims[0])]
            self.value = Matrix(dims[0], dims[1], bits, value=rand)
        elif isinstance(value, pyrtl.Input):
            self.value = Matrix(dims[0], dims[1], bits, value=value)
        else:
            self.value = value
        self.rows = dims[0]
        self.columns = dims[1]

        # auto grad related properties
        self.requires_grad = False
        self.grad = None
        self.grad_fn = None
        self.is_leaf = True

    @staticmethod
    def argmax(tensor: HDLTensor, axis: int) -> HDLTensor:
        mat = matrix.argmax(tensor.value, axis=axis)
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
        temp = self.value
        for i in range(self.rows):
            for j in range(self.columns):
                temp[i, j] = temp[i, j] // b.value[i % b.rows, j % b.columns]

    def __truediv__(self, b: HDLTensor) -> HDLTensor:
        '''
        Perform the element-wise division op.
        '''
        return self.__div__(b)

    def reshape(self, *newshape, **order) -> HDLTensor:
        return self.value.reshape(self.value, *newshape, **order)

    def ternary(self, row: int, col: int, cond_fn: function, true, false) -> None:
        self.value[row, col] = select(
            cond_fn(self.value[row, col]), true, false
        )

    def to_wirevector(self) -> Matrix:
        return self.value.to_wirevector()

    def __matmul__(self, other: HDLTensor) -> HDLTensor:
        mat = self.value @ other.value
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def __imatmul__(self, other):
        self.value @= other.value
        return self

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
        return HDLTensor([
            self.rows, self.columns],
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
        self.value = self.value.transpose()


if __name__ == '__main__':
    tensor = HDLTensor(30, 30)
    print(tensor[0, 0])
    print(tensor[1, 1])
