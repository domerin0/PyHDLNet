from __future__ import annotations
import random
import pyrtl
from pyrtl.rtllib.matrix import Matrix, WireVector, Const
import pyrtl.rtllib.matrix as matrix
from pyrtl.corecircuits import select, shift_left_logical
from pyrtl.corecircuits import shift_left_arithmetic
from pyrtl.corecircuits import shift_right_arithmetic, shift_right_logical
from utility import uniform_int, apply_tensor, clamp
from utility import reduce_tensors, min_tensor, max_tensor, apply_tensor_per_channel
from typing import Any


class HDLScalar:
    def __init__(self, value, bits=None, signed=False):
        self.value = Const(value, bitwidth=bits, signed=signed)
        self.bits = bits


class HDLTensor:
    def __init__(self, *dims, value=None, bits=8, signed=False) -> None:
        if len(dims) > 2:
            raise ValueError("HDLTensor only supports 2D matrices")
        self.signed = signed
        self.shape = dims
        self.original_bits = bits

        if signed:
            self.max_int = 2 ** (bits - 1) - 1
            self.min_int = -(2 ** (bits - 1))
        else:
            self.max_int = 2 ** bits - 1
            self.min_int = 0

        if value is None:
            rand = uniform_int(self.max_int, self.min_int, dims[0], dims[1])
            self.value = Matrix(dims[0], dims[1], bits,
                                value=rand, max_bits=2*bits)
        elif isinstance(value, pyrtl.Input) or isinstance(value, list):
            self.value = Matrix(dims[0], dims[1], bits,
                                value=value, max_bits=2*bits)
        elif isinstance(value, str) and value == 'input':
            self.value = Matrix(
                dims[0],
                dims[1],
                bits,
                value=pyrtl.Input(
                    bitwidth=dims[0]*dims[1]*bits,
                    name='net_input'),
                max_bits=2*bits
            )
        else:
            self.value = value
        self.acc = Matrix(
            1, 3,
            bits, value=[[0, 0, 0]],
            max_bits=2*bits
        )
        self.one = Const(1, bitwidth=bits)
        # self.acc3 = Matrix(1, 1, bits,
        #                        value=[[0]], max_bits=bits)
        # self.acc2 = Matrix(1, 1, bits,
        #                         value=[[0]], max_bits=bits)
        self.shift_left = shift_left_logical
        self.shift_right = shift_right_logical
        if self.signed:
            self.shift_left = shift_left_arithmetic
            self.shift_right = shift_right_arithmetic

        # auto grad related properties
        self.requires_grad = False
        self.grad = None
        self.grad_fn = None
        self.is_leaf = True

    def argmax(self, axis: int) -> HDLTensor:
        mat = matrix.argmax(self.value, axis=axis)
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    @staticmethod
    def multiply(first: HDLTensor, second: Any) -> HDLTensor:
        '''
        Performs element-wise or scale multipliation of two tensors.

        Args:
            first (HDLTensor): The first tensor to multiply.
            second (Any): The second tensor to multiply or scalar.
        Returns:
            HDLTensor: A tensor containing the result of the multiplication.
        '''
        mat = matrix.multiply(first.value, second.value)

        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def sum(self, axis: int) -> HDLTensor:
        '''
        Sums the tensor along the specified axis.

        Args:
            axis (int): The axis to sum along. 0 is columns 1 is rows. None is the entire matrix.
        Returns:
            HDLTensor: A tensor containing the sum along the specified axis.

        '''
        mat = matrix.sum(self.value, axis=axis)
        return HDLTensor(mat.rows, mat.columns, bits=mat.bits, value=mat)

    def max(self, axis: int):
        '''
        Gets the maximum value of the tensor along the specified axis.

        Args:
            axis (int): The axis to get the maximum value from. 0 is columns
                1 is rows. None is the entire matrix.
        Returns:
            HDLTensor: A tensor containing the maximum value along the specified axis.
        '''
        val = matrix.max(self.value, axis=axis)
        return HDLTensor(val.rows, val.columns, bits=val.bits, value=val)

    def truncate_lsb(self, msb: int, acc_bits: int):
        '''
        Truncates the least significant bits of the tensor.
        Keeps only msb bits from the tensor.

        Args:
            msb (int): The number of bits to keep from the most significant bit.
            acc_bits (int): The number of useful accumulated bits.
        '''
        for i in range(self.rows):
            for j in range(self.columns):
                self.value[i, j] = self.value[i, j][msb-1:acc_bits]
            # self.value.bits = bits

    def add_inplace(self, other: WireVector) -> HDLTensor:
        '''
        Element-wise inplace add.

        Args:
            other (HDLTensor): The other tensor to add.
        Returns:
            HDLTensor: A tensor containing the result of the addition.
        '''
        for r in range(self.rows):
            for c in range(self.columns):
                self.value[r, c] = self.value[r, c] + other
        return self

    def divide_nbit_unsigned(
        self,
        dividend: WireVector,
        divisor: WireVector,
        bits: int,
    ):
        '''
            Performs unsigned division of two n-bit numbers.
            Inputs:
                bits: Number of bits in the dividend and divisor
                dividend: Dividend
                divisor: Divisor
        '''
        temp_acc = self.acc.copy()
        # Initialize quotient as 0
        temp_acc[0, 0] = 0

        # Initialize remainder as 0
        temp_acc[0, 1] = 0

        # Initialize last remainder value as 0
        temp_acc[0, 2] = 0

        # Iterate through each bit
        for i in range(bits):

            # Left-shift remainder and add next bit of num
            temp_acc[0, 1] = self.shift_left(temp_acc[0, 1], 1) | (
                self.shift_right(dividend, bits - 1 - i) & self.one)

            # last remainder
            temp_acc[0, 2] = temp_acc[0, 1]

            temp_acc[0, 1] = select(
                # remainder greater than divisor
                temp_acc[0, 2] >= divisor,
                temp_acc[0, 1] - divisor,
                temp_acc[0, 1]
            )

            if i == bits - 1:
                val = self.one
            else:
                val = self.shift_left(self.one, bits - 1 - i)

            temp_acc[0, 0] = select(
                # remainder greater than divisor
                temp_acc[0, 2] >= divisor,
                temp_acc[0, 0] | val,
                temp_acc[0, 0]
            )

        # return quotient
        return temp_acc[0, 0]

    def log2(self, row: int, col: int):
        '''
        Calculates the log2 of the tensor at the specified row and column.
        '''
        # Initialize accumulator
        self.acc[0, 0] = self.value[row, col]

        # Initialize counter
        self.acc[0, 1] = 0

        for _ in range(self.bits):
            self.acc[0, 0] = self.shift_right(self.acc[0, 0], 1)
            self.acc[0, 1] = select(
                self.acc[0, 0] > 0,
                self.acc[0, 1] + self.one,
                self.acc[0, 1]
            )
        return self.acc[0, 1]

    def divide_scalar(self, divisor: WireVector, row: int, col: int):
        return self.divide_nbit_unsigned(
            dividend=self[row, col],
            divisor=divisor,
            bits=self.bits,
        )

    def divide(self, divisor: HDLTensor, acc: HDLTensor,
               row: int = None, col: int = None) -> HDLTensor:
        '''
        Performs element-wise division of two tensors.
        Also does scalar division if divisor is 1x1 tensor.

        Args:
            dividend (HDLTensor): The dividend.
            divisor (HDLTensor): The divisor.
            acc (HDLTensor): The accumulator.
        '''
        if divisor.rows != self.rows or divisor.columns != self.columns:
            if divisor.rows != 1 or divisor.columns != 1:
                raise ValueError(
                    "Cannot divide HDLTensor of different dimensions"
                )
        for dividend_row in range(self.rows):
            for dividend_col in range(self.columns):
                dividend_val = self.value[dividend_row, dividend_col]

                if divisor.value.rows > 1:
                    divisor_val = divisor.value[dividend_row, dividend_col]
                else:
                    divisor_val = divisor.value[0, 0]

                acc.value[dividend_row, dividend_col] = self.divide_nbit_unsigned(
                    dividend=dividend_val,
                    divisor=divisor_val,
                    bits=divisor.bits,
                )

        return acc

    def __floordiv__(self, b: HDLTensor) -> HDLTensor:
        '''
        Perform the element-wise division op.
        b must match the dimensions of self or be a scalar (1x1 tensor). 
        '''
        raise NotImplementedError("Use divide() instead")

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
        '''
            Returns the number of bits in the tensor.
        '''
        return self.value.bits

    @property
    def columns(self):
        '''
            Returns the number of columns in the tensor.
        '''
        return self.value.columns

    @property
    def rows(self):
        '''
            Returns the number of rows in the tensor.
        '''
        return self.value.rows

    def to_wirevector(self):
        return self.value.to_wirevector()

    def __matmul__(self, other: HDLTensor) -> HDLTensor:
        '''
        Performs matrix multiplication of two tensors.
        '''
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
        if isinstance(other, int) or isinstance(other, WireVector):
            return HDLTensor(
                self.rows, self.columns,
                bits=self.bits,
                value=self.value + other
            )
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
                result[i, j] = self.shift_left(self[i, j], 1)
        return result

    def __iadd__(self, other):
        self = self.__add__(other)
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

    def __getitem__(self, *tup):
        x, y = tup[0]
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


class Tensor:
    def __init__(self,
                 *dims,
                 value=None) -> None:
        '''
        This is just a data holding class we can import into 
        pure python code as it doesn't rely on the PyRTL library.
        '''
        if len(dims) > 2:
            raise ValueError("PyTensor only supports 2D matrices")

        if value is None:
            rand = uniform_int(255, 0, dims[0], dims[1])
            self.value = rand
        else:
            self.value = value
        self.dims = dims
        self.rows = dims[0]
        self.columns = dims[1]

    @property
    def shape(self):
        return self.dims

    def __add__(self, other):
        if isinstance(other, Tensor):
            tensor = reduce_tensors(
                self.value, other.value, lambda x, y: x + y)
        else:
            tensor = apply_tensor(
                self.value, lambda x: x + other)
        return Tensor(*self.dims, value=tensor)

    def __iadd__(self, other):
        self.value = self.__add__(other)
        return self

    def __mul__(self, other):
        if isinstance(other, Tensor):
            tensor = reduce_tensors(
                self.value, other.value, lambda x, y: x * y)
        else:
            tensor = apply_tensor(
                self.value, lambda x: x * other)
        return Tensor(*self.dims, value=tensor)

    def __imul__(self, other):
        self.value = self.__mul__(other)
        return self

    def __sub__(self, other):
        if isinstance(other, Tensor):
            tensor = reduce_tensors(
                self.value, other.value, lambda x, y: x - y)
        else:
            tensor = apply_tensor(
                self.value, lambda x: x - other)
        return Tensor(*self.dims, value=tensor)

    def __isub__(self, other):
        self.value = self.__sub__(other)
        return self

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            tensor = reduce_tensors(
                self.value, other.value, lambda x, y: x / y)
        else:
            tensor = apply_tensor(
                self.value, lambda x: x / other)
        return Tensor(*self.dims, value=tensor)

    def clamp(self, u, l):
        tensor = apply_tensor(
            self.value, lambda x: clamp(x, u, l))
        return Tensor(*self.dims, value=tensor)

    def __floordiv__(self, other):
        if isinstance(other, Tensor):
            tensor = reduce_tensors(
                self.value, other.value, lambda x, y: x // y)
        else:
            tensor = apply_tensor(
                self.value, lambda x: x // other)
        return Tensor(*self.dims, value=tensor)

    def __mod__(self, other):
        if isinstance(other, Tensor):
            tensor = reduce_tensors(
                self.value, other.value, lambda x, y: x % y)
        else:
            tensor = apply_tensor(
                self.value, lambda x: x % other)
        return Tensor(*self.dims, value=tensor)

    def max(self, channel=None):
        return max_tensor(self.value, channel=channel)

    def min(self, channel=None):
        return min_tensor(self.value, channel=channel)

    def round(self):
        tensor = apply_tensor(
            self.value, lambda x: round(x))
        return Tensor(*self.dims, value=tensor)

    def __getitem__(self, dims):
        temp = self.value
        for d in dims:
            if not isinstance(temp, list):
                raise ValueError(
                    "PyTensor value for dim {0} is not a Tensor"
                ).format(d)
            temp = temp[d]
        return temp

    def transpose(self, dim1, dim2):
        raise NotImplementedError("Transpose not implemented yet")

    def __setitem__(self, dims, value):
        temp = self.value
        working = self.value
        for d in dims:
            if not isinstance(temp, list):
                raise ValueError(
                    "PyTensor value for dim {0} is not a Tensor"
                ).format(d)
            if not isinstance(working[d], list) and d == dims[-1]:
                working[d] = value
            working = working[d]

        temp[dims[-1]] = value

    def quantize_per_channel(self, bits=8, signed=False):
        '''
        Per-Channels quantization of a tensor

        Implementation not done yet (don't use this)
        '''
        max_int = 2**(bits - 1) - 1 if signed else 2**bits - 1
        min_int = -2**(bits - 1) if signed else 0

        quantized = []
        max_vals = []
        min_vals = []

        for channel in range(self.dims[-1]):
            max_vals.append(self.max(channel=channel))
            min_vals.append(self.min(channel=channel))
        for row_i in range(self.dims[0]):
            val_channel = self.value[row_i]

            scaled_weights = apply_tensor_per_channel(
                val_channel,
                lambda i, x: clamp(
                    round(
                        ((x - min_vals[i]) / (max_vals[i] - min_vals[i])) * max_int),
                    max_int, min_int
                ))

            quantized.append(scaled_weights)

        return Tensor(*self.dims, value=quantized)

    def quantize(self, bits=8, signed=False, symmetric=False):
        min_val = self.min()
        max_val = self.max()

        max_int = 2**(bits - 1) - 1 if signed else 2**bits - 1
        if symmetric:
            min_int = -2**(bits - 1) - 1 if signed else 0
        else:
            min_int = -2**(bits - 1) if signed else 0

        def q_scalar(x):
            return clamp(round(((x - min_val) / (max_val - min_val)) * max_int), max_int, min_int)

        tensor = apply_tensor(
            self.value, lambda x: q_scalar(x))
        return Tensor(*self.dims, value=tensor)

    def tolist(self):
        return self.value


0


def product(arr):
    i = 1
    for x in arr:
        i *= x
    return i


class TestTensor:
    def __init__(self, arr=[]):
        self.__arr = self.process(arr)
        self.dimensions = self.get_dims()

    def __len__(self):
        return len(self.__arr)

    def process(self, arr):
        new_list = []  # make a new array
        for i in arr:
            if type(i) == list:  # make arrays new tensors, add those new tensors to the array
                new_list.append(Tensor(i))
            else:
                new_list.append(i)  # add numbers to the 1D tensors
        return new_list

    def quantize(self, bits=8, signed=False, symmetric=False):
        min_val = self.min()
        max_val = self.max()

        max_int = 2**(bits - 1) - 1 if signed else 2**bits - 1
        if symmetric:
            min_int = -2**(bits - 1) - 1 if signed else 0
        else:
            min_int = -2**(bits - 1) if signed else 0

        def q_scalar(x):
            return clamp(round(((x - min_val) / (max_val - min_val)) * max_int), max_int, min_int)

        tensor = apply_tensor(
            self.value, lambda x: q_scalar(x))
        return Tensor(tensor)

    def get_dims(self):
        dims = []  # array that will contain dimension numbers ([outer, inner])
        dims.append(len(self.__arr))  # get the length of the processed array
        if type(self.__arr[0]) == Tensor:
            # add on the dimension of the contained tensor
            dims += self.__arr[0].get_dims()
        return dims

    def __getitem__(self, ind):
        return self.__arr[ind]

    def tolist(self):
        arr = []  # the full array
        # see if the tensor is contains numbers or tensors
        isVector = len(self.dimensions) == 1
        for i in self.__arr:
            if isVector:  # append the number if it's a number
                arr.append(i)
            else:  # append the array made by the tensor (i)
                # add the array that that tensor represents
                arr.append(i.tolist())
        return arr

    def to_1d_vec(self):
        vector = []

        isVector = len(self.dimensions) == 1
        for i in self.__arr:
            if isVector:
                vector.append(i)
            else:
                vector += i.to_1d_vec()

        return vector

    def package(arr, dims):
        if len(arr) == 1:
            return arr[0]

        new_list = []

        index = 0
        p = product(dims[1:])
        for i in range(dims[0]):
            new_list.append(Tensor.package(arr[index:index+p], dims[1:]))
            index += p

        return Tensor(new_list)

    def zeros(*dims):
        arr = [0] * product(dims)
        if len(dims) == 1:
            return Tensor(arr)
        return Tensor.package(arr, dims)

    def rand(*dims):
        arr = []
        for i in range(product(dims)):
            arr.append(random.random())
        if len(dims) == 1:
            return Tensor(arr)
        return Tensor.package(arr, dims)

    def transpose(self, order=[]):
        if len(self.dimensions) == 1:
            return Tensor(self.__arr)
        if not order:
            order = list(reversed(range(len(self.dimensions))))

        new_dims = [self.dimensions[i] for i in order]
        # make an array (to be a tensor) with the dimensions of the new tensor, all spots are defaulted to 0
        arr = Tensor.zeros(*new_dims).tolist()
        # GROUP 3
        # go through all possible paths in the tensor
        paths = [0]*len(self.dimensions)
        while paths[0] < self.dimensions[0]:
            # get references to the path, put the number in the tensor to its corresponding spot in the new tensor
            ref = self
            place = arr
            for i in range(len(paths) - 1):
                ref = ref[paths[i]]
                place = place[paths[order[i]]]
            place[paths[order[-1]]] = ref[paths[-1]]
            # GROUP 4
            # go to the next path (sequentially)
            paths[-1] += 1
            for i in range(len(paths)-1, 0, -1):
                if paths[i] >= self.dimensions[i]:
                    paths[i] = 0
                    paths[i-1] += 1
                else:
                    break
        return Tensor(arr)

    def __add__(self, other):
        new_list = []

        if type(other) != Tensor:
            for i in self.__arr:
                new_list.append(i+other)

        else:
            for i in range(len(self.__arr)):
                new_list.append(self.__arr[i] + other[i])

        return Tensor(new_list)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        new_list = []  # create a new tensor

        if type(other) != Tensor:  # it the thing being added is a scalar, add it to all elements
            for i in self.__arr:
                new_list.append(i*other)

        else:  # if a tensor is being added, add the corresponding elements of the two tensors
            for i in range(len(self.__arr)):
                new_list.append(self.__arr[i] * other[i])

        return Tensor(new_list)

    def __truediv__(self, other):
        new_list = []  # create a new tensor

        if type(other) != Tensor:  # it the thing being added is a scalar, add it to all elements
            for i in self.__arr:
                new_list.append(i/other)

        else:  # if a tensor is being added, add the corresponding elements of the two tensors
            for i in range(len(self.__arr)):
                new_list.append(self.__arr[i] / other[i])

        return Tensor(new_list)

    def __floordiv__(self, other):
        new_list = []  # create a new tensor

        if type(other) != Tensor:  # it the thing being added is a scalar, add it to all elements
            for i in self.__arr:
                new_list.append(i/other)

        else:  # if a tensor is being added, add the corresponding elements of the two tensors
            for i in range(len(self.__arr)):
                new_list.append(self.__arr[i] // other[i])

        return Tensor(new_list)

    def __matmul__(self, other):

        def firstVector(v, m):
            v = Tensor([v])
            return Tensor.matmul(v, m)[0]

        def secondVector(m, v):
            v = Tensor([[v[i]] for i in range(len(v))])
            r = Tensor.matmul(m, v)
            return Tensor([r[i][0] for i in range(len(r))])

        def matrices(m1, m2):

            new_list = []

            for _ in range(m1.dimensions[0]):
                new_list.append([0]*m2.dimensions[1])

            # for all columns in the second matrix
            for i in range(m2.dimensions[1]):
                # for all rows in the first matrix
                for j in range(m1.dimensions[0]):
                    # for all numbers in the row of the first matrix
                    new_list[j][i] = sum([m2[k][i] * m1[j][k]
                                          for k in range(m1.dimensions[1])])

            return Tensor(new_list)

        def nd(nm1, nm2):  # treat as a stack of matrices
            new_list = []
            if len(nm1.dimensions) > 2:
                for i in range(nm1.dimensions[0]):
                    new_list.append(nm1[i].matmul(nm2))
                return Tensor(new_list)

            elif len(nm2.dimensions) > 2:
                for i in range(nm2.dimensions[0]):
                    new_list.append(nm1.matmul(nm2[i]))
                return Tensor(new_list)

        if len(self.dimensions) == 1 and len(other.dimensions) == 2:
            return firstVector(self, other)
        elif len(self.dimensions) == 2 and len(other.dimensions) == 1:
            return secondVector(self, other)
        elif len(self.dimensions) == 2 and len(other.dimensions) == 2:
            return matrices(self, other)
        else:
            return nd(self, other)

    def __add__(self, other):
        if other.dimensions[1:] == self.dimensions:
            return Tensor(self.tolist() + other.tolist())

    def __mul__(self, other):
        new_list = []
        for i in range(len(self)):
            for j in range(len(other)):
                if len(self.dimensions) == 1 or len(other.dimensions) == 1:
                    new_list.append(self[i] * self[j])
                else:
                    new_list.append(self[i].product(self[j]))

        return Tensor.package(new_list, [len(self), len(other)])

    def __neg__(self):
        return self * -1

    def __str__(self):
        ret = "\n["
        commas = False
        if type(self.__arr[0]) is not Tensor:
            commas = True
        for i in self.__arr:
            ret += str(i)
            if commas:
                ret += ', '

        if commas:
            ret = ret[:-2]
        else:
            ret += '\n'
        ret += ']'

        for i in range(len(ret) - 2, 0, -1):
            if ret[i] == '\n' and ret[i+1] in '[]' and ret[i-1] == ret[i+1]:
                ret = ret[:i] + ret[i+1:]

        return ret

    def __repr__(self):
        ret = "\n["
        commas = False
        if type(self.__arr[0]) is not Tensor:
            commas = True
        for i in self.__arr:
            ret += str(i)
            if commas:
                ret += ', '

        if commas:
            ret = ret[:-2]
        else:
            ret += '\n'
        ret += ']'

        for i in range(len(ret) - 2, 0, -1):
            if ret[i] == '\n' and ret[i+1] in '[]' and ret[i-1] == ret[i+1]:
                ret = ret[:i] + ret[i+1:]

        return ret
