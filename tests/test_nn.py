import unittest
from nn import Linear, SoftMax, ReLU
from tensor import HDLTensor
import pyrtl
from utils import matrix_list_equal

bits = 8
signed = False


class TestNeuralNetworkLayers(unittest.TestCase):
    def setUp(self):
        pyrtl.reset_working_block()
        # pyrtl.set_debug_mode(True)

    def test_linear_forward(self):
        linear_1 = Linear(
            in_d=3, out_d=4,
            weights=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]],
            bias=[[1, 2, 5, 3]],
            bits=bits, signed=signed
        )
        x = HDLTensor(2, 3, value=[[100, 50, 231], [200, 143, 234]])

        output = linear_1.forward(x)

        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            output.value,
            [[101, 52, 236, 53], [201, 145, 239, 146]]
        ))

    def test_softmax_forward(self):
        softmax = SoftMax()
        softmax.eval()
        x = HDLTensor(2, 3, value=[[100, 50, 231], [200, 9, 143]])

        output = softmax.forward(x)
        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            output.value,
            [[2, 0]]
        ))

    def test_relu_forward(self):
        relu = ReLU(zero_point=2**8//2)
        relu.eval()
        x = HDLTensor(2, 3, value=[[100, 50, 231], [200, 9, 143]])

        output = relu.forward(x)
        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            output.value,
            [[128, 128, 231], [200, 128, 143]]
        ))


if __name__ == '__main__':
    unittest.main()
