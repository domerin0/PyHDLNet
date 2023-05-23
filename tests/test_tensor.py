import unittest
from tensor import HDLTensor, HDLScalar
import pyrtl
from utils import matrix_list_equal
from math import log
import random

bits = 8
signed = False


class TestHDLTensor(unittest.TestCase):

    def setUp(self):
        pyrtl.reset_working_block()

    def test_unsigned_scalar_divide(self):
        x = HDLTensor(
            1, 256, value=[[i for i in range(256)]], bits=bits, signed=signed)
        y = HDLTensor(1, 1, value=[[3]], bits=bits, signed=signed)
        acc = HDLTensor(1, 256, bits=bits, signed=signed)

        result = x.divide(y, acc=acc)

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[i // 3 for i in range(256)]]
        ))

    def test_truncate_lsb(self):
        rand_bits = random.randint(9, 64)
        int_val = random.randint(0, 2**rand_bits - 1)
        x = HDLTensor(
            1, 1, value=[[int_val]], bits=rand_bits, signed=signed)

        x.truncate_lsb(bits)
        self.assertTrue(matrix_list_equal(
            x.value,
            [[int_val >> (rand_bits - bits)]]
        ))

    def test_unsigned_scalar_floor_divide(self):
        x = HDLTensor(
            1, 256, value=[[i for i in range(256)]], bits=bits, signed=signed)
        y = HDLScalar(3)

        result = x // y

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[i // 3 for i in range(256)]]
        ))

    def test_unsigned_scalar_true_divide(self):
        x = HDLTensor(
            1, 256, value=[[i for i in range(256)]], bits=bits, signed=signed)
        y = HDLScalar(3)

        result = x / y

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[i // 3 for i in range(256)]]
        ))

    def test_unsigned_vector_floor_divide(self):
        x = HDLTensor(
            1, 3, value=[[3, 25, 87]], bits=bits, signed=signed)
        y = HDLTensor(1, 3, value=[[16, 16, 17]], bits=bits, signed=signed)

        result = x // y

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[3 // 16, 25 // 16, 87 // 17]]
        ))

    def test_unsigned_vector_true_divide(self):
        x = HDLTensor(
            1, 3, value=[[3, 25, 87]], bits=bits, signed=signed)
        y = HDLTensor(1, 3, value=[[16, 16, 17]], bits=bits, signed=signed)

        result = x / y

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[3 // 16, 25 // 16, 87 // 17]]
        ))

    def test_unsigned_vector_divide(self):
        x = HDLTensor(
            1, 3, value=[[3, 25, 87]], bits=bits, signed=signed)
        y = HDLTensor(1, 3, value=[[16, 16, 17]], bits=bits, signed=signed)
        acc = HDLTensor(1, 3, bits=bits, signed=signed)

        result = x.divide(y, acc=acc)

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[3 // 16, 25 // 16, 87 // 17]]
        ))

    def test_log2(self):
        x = HDLTensor(
            1, 255, value=[[i for i in range(1, 256)]], bits=bits, signed=signed)

        for i in range(x.rows):
            for j in range(x.columns):
                x.value[i, j] = x.log2(i, j)

        self.assertTrue(matrix_list_equal(
            x.value,
            [[round(int(log(i, 2))) for i in range(1, 256)]]
        ))

    def test_matmul(self):
        x = HDLTensor(3, 1, value=[[2], [2], [2]])
        y = HDLTensor(1, 3, value=[[2, 2, 2]])

        result = x @ y

        self.assertEqual(isinstance(result, HDLTensor), True)
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.columns, 3)
        self.assertTrue(matrix_list_equal(
            result.value,
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]]
        ))


if __name__ == '__main__':
    unittest.main()
