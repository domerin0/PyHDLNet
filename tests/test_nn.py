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

    def test_linear_backward_output(self):
        linear_1 = Linear(
            in_d=3, out_d=4,
            weights=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]],
            bias=[[1, 2, 5, 3]],
            bits=bits, signed=signed
        )
        linear_1.train()

        x = HDLTensor(
            2, 3,
            signed=signed, bits=bits,
            value=[[100, 50, 231], [200, 143, 234]]
        )

        d_out = HDLTensor(
            2, 4,
            bits=bits, signed=signed,
            value=[[163, 20, 33, 50], [255, 200, 39, 104]]
        )

        output = linear_1.backward(d_out, x)

        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            output.value,
            [
                [163,  70,  33],
                [255, 304,  39]
            ]
        ))

    def test_linear_backward_grad_w(self):
        linear_1 = Linear(
            in_d=3, out_d=4,
            weights=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]],
            bias=[[1, 2, 5, 3]],
            truncate_bits=bits,
            bits=bits, signed=signed
        )
        linear_1.train()

        x = HDLTensor(
            2, 3,
            signed=signed, bits=bits,
            value=[[100, 50, 231], [200, 143, 234]]
        )

        d_out = HDLTensor(
            2, 4,
            bits=bits, signed=signed,
            value=[[163, 20, 33, 50], [255, 200, 39, 104]]
        )

        output = linear_1.backward(d_out, x)

        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            linear_1.grad_w.value,
            [[175, 109, 28, 67], [116, 77, 18, 45], [253, 133, 43, 93]]
        ))

    def test_linear_backward_grad_b(self):
        linear_1 = Linear(
            in_d=3, out_d=4,
            weights=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]],
            bias=[[1, 2, 5, 3]],
            bits=bits, signed=signed
        )
        linear_1.train()

        x = HDLTensor(
            2, 3,
            signed=signed, bits=bits,
            value=[[100, 50, 231], [200, 143, 234]]
        )

        d_out = HDLTensor(
            2, 4,
            bits=bits, signed=signed,
            value=[[163, 20, 33, 50], [255, 200, 39, 104]]
        )

        output = linear_1.backward(d_out, x)

        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            linear_1.grad_b.value,
            [[139,  73, 24, 51]]
        ))

    def test_softmax_eval_forward(self):
        softmax = SoftMax()
        softmax.eval()
        x = HDLTensor(2, 3, value=[[100, 50, 231], [200, 9, 143]])

        output = softmax.forward(x)
        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            output.value,
            [[2, 0]]
        ))

    def test_softmax_train_forward(self):
        softmax = SoftMax()
        softmax.train()
        x = HDLTensor(2, 3, value=[[100, 50, 231], [200, 9, 143]])

        output = softmax.forward(x)
        self.assertEqual(isinstance(output, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            output.value,
            [[39, 9, 207], [169, 0, 86]]
        ))

    def test_relu_forward(self):
        relu = ReLU()
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
