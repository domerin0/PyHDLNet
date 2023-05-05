import unittest
from transforms import *
from dataloaders import Dataset


class TestDataLoader(Dataset):
    def __init__(self):
        self.dataset = {"data": []}
        for i in range(100):
            self.dataset["data"].append(([str(i), str(i+1)], i))


class TestTransforms(unittest.TestCase):

    def test_float_transform(self):
        test_set = TestDataLoader()
        transform = FloatTransform([1])
        data = transform(test_set)
        self.assertEqual(len(data["data"]), 100)
        for row in data["data"]:
            self.assertTrue(isinstance(row[0][0], str))
            self.assertTrue(isinstance(row[0][1], float))


if __name__ == '__main__':
    unittest.main()
