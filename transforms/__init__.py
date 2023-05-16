from typing import Any, List
import random
from abc import ABC, abstractmethod
from dataloaders import Dataset, TransformedDataset
import math
from tensor import Tensor


class NeuralTransform(ABC):
    @abstractmethod
    def __call__(self, data: Dataset) -> Dataset:
        pass


class TransformsPipeline:
    def __init__(self, transforms: List[NeuralTransform]):
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        return self.transforms[index]

    def __call__(self, data: Dataset):
        for transform in self.transforms:
            data = transform(data)
        return data


class FloatTransform(NeuralTransform):
    def __init__(self, indices):
        self.indices = indices

    def __call__(self, data: Dataset) -> Dataset:
        for key in list(data.get_names()):
            data[key] = self.floatify(data[key])
        return data

    def floatify(self, data: List[List[object]]):
        for i in range(len(data)):
            for index in self.indices:
                data[i][0][index] = float(data[i][0][index])
        return data


class DataShufflerTransform(NeuralTransform):
    def __init__(self):
        pass

    def __call__(self, data: Dataset) -> Dataset:
        for key in list(data.get_names()):
            random.shuffle(data[key])
        return data


class DataSplitterTransform(NeuralTransform):
    def __init__(self, names: List[str], ratios: List[float]):
        for ratio in ratios:
            if ratio < 0 or ratio > 1:
                raise ValueError(
                    "Ratio {0} must be between 0 and 1".format(ratio))
        if len(names) != len(ratios):
            raise ValueError(
                "Names {0} and ratios {1} must be same length"
                .format(len(names), len(ratios))
            )
        if sum(ratios) != 1:
            raise ValueError(
                "Ratios {0} must sum to 1".format(ratios))
        self.names = names
        self.ratios = ratios

    def __call__(self, data: Dataset) -> Dataset:
        index = 0
        dic = {}
        dataset = []
        for key in list(data.get_names()):
            dataset += data[key]
        for i in range(len(self.ratios)):
            num_items = math.floor(len(dataset) * self.ratios[i])
            dic[self.names[i]] = dataset[index: num_items + index]
            index += num_items
        return TransformedDataset(dic)


class QuantizeTransform(NeuralTransform):
    def __init__(self,
                 bits: int,
                 signed: bool = True,
                 symmetric=True):
        if not isinstance(bits, int):
            raise ValueError("Bits must be an integer")
        self.bits = bits
        self.symmetric = symmetric
        self.signed = signed
        if signed:
            self.max_int = (2 ** (bits - 1)) - 1
            self.min_int = -(2 ** (bits - 1)) if not symmetric else \
                -((2 ** (bits - 1)) - 1)
        else:
            self.max_int = (2 ** bits) - 1
            self.min_int = 0

    def __call__(self, data: Dataset) -> Dataset:
        for key in list(data.get_names()):
            matrix = [r[0] for r in data[key]]
            tensor = Tensor(
                len(data[key]), len(data[key][0]),
                value=matrix
            )
            matrix = tensor.quantize(
                bits=self.bits, signed=self.signed
            ).tolist()
            for i in range(len(matrix)):
                data[key][i] = (matrix[i], data[key][i][1])
        return data


class NormalizeZScoreTransform(NeuralTransform):
    def __init__(self, means, stds):
        if len(means) != len(stds):
            raise ValueError(
                "Means {0} and stds {1} must be same length"
                .format(len(means), len(stds))
            )
        self.means = means
        self.stds = stds

    def normalize(self, data: List):
        for row in data:
            for i in range(len(row)):
                row[0][i] = (float(row[0][i]) - self.means[i]) / self.stds[i]
        return data

    def __call__(self, data: Dataset) -> Dataset:
        for key in list(data.get_names()):
            data[key] = self.normalize(data[key])
        return data
