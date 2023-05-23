import unittest
from criterion import CrossEntropyLoss
from tensor import HDLTensor
import pyrtl
from utils import matrix_list_equal

bits = 8
signed = False


class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        pyrtl.reset_working_block()
        # pyrtl.set_debug_mode(True)

    def test_cross_entropy_loss_8bits(self):
        ce_loss = CrossEntropyLoss()
        y_true = HDLTensor(1, 3, value=[[255, 0, 0]])
        y_pred = HDLTensor(1, 3, value=[[51, 100, 104]])

        loss = ce_loss(y_pred=y_pred, y_true=y_true)

        self.assertEqual(isinstance(loss, HDLTensor), True)
        self.assertTrue(matrix_list_equal(
            loss.value,
            [[1275]]
        ))


if __name__ == '__main__':
    unittest.main()
