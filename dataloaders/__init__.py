import csv
from abc import ABC
from tensor import PyTensor
from typing import List, Any


class Dataset(ABC):
    def __getitem__(self, name):
        return self.dataset[name]

    def __setitem__(self, name, value):
        self.dataset[name] = value

    def __len__(self):
        keys = list(self.dataset.keys())
        count = 0
        for key in keys:
            count += len(self.dataset[key])

        return count

    def get_names(self):
        return list(self.dataset.keys())


class TransformedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, name):
        return self.dataset[name]

    def __setitem__(self, name, value):
        self.dataset[name] = value

    def __len__(self):
        keys = list(self.dataset.keys())
        count = 0
        for key in keys:
            count += len(self.dataset[key])

        return count

    def get_names(self):
        return list(self.dataset.keys())


class ListLoader(Dataset):
    def __init__(self, tensor: List[List[Any]], label_index=-1):
        self.tensor = tensor
        transformed = {"data": []}

        for row in tensor:
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                except:
                    pass
            label = row[label_index]
            features = row
            del features[label_index]
            transformed["data"].append((features, label))
        self.transformed = transformed

    def __getitem__(self, name):
        return self.transformed[name]

    def __setitem__(self, name, value):
        self.transformed[name] = value

    def __len__(self):
        keys = list(self.tensor.keys())
        count = 0
        for key in keys:
            count += len(self.transformed[key])

        return count

    def get_names(self):
        return list(self.transformed.keys())


class CSVLoader(Dataset):
    def __init__(self, file_path, has_header=True, label_index=-1):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)

        if has_header:
            data = data[1:]

        transformed = {"data": []}

        for row in data:
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                except:
                    pass
            label = row[label_index]
            features = row
            del features[label_index]
            transformed["data"].append((features, label))

        self.dataset = transformed
