import unittest
from dataloaders import CSVLoader


class TestCSVDataLoader(unittest.TestCase):

    def test_len(self):
        data = CSVLoader("data.csv")
        self.assertEqual(len(data), 150)

    def test_getnames(self):
        data = CSVLoader("data.csv")
        self.assertEqual(data.get_names(), ["data"])


if __name__ == '__main__':
    unittest.main()
