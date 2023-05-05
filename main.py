from nn import Sequential, Linear, ReLU, SoftMax
from tensor import HDLTensor
import pyrtl
import pyrtl.rtllib.matrix as matrix
from logic_statistics import print_stats
from transforms import TransformsPipeline, DataShufflerTransform, DataSplitterTransform, NormalizeZScoreTransform, QuantizeTransform
from dataloaders import CSVLoader
from plot import plot

num_bits = 16
input_size = 4


if __name__ == '__main__':
    a_in = pyrtl.Input(input_size * num_bits, 'a_in')

    a = HDLTensor(1, 4, bits=num_bits, value=a_in)

    dataset = CSVLoader("data.csv", has_header=True)
    dataset2 = CSVLoader("data.csv", has_header=True)

    transforms = TransformsPipeline([
        NormalizeZScoreTransform(
            [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]
        ),
        QuantizeTransform(bits=8),
        DataSplitterTransform(['train', 'test'], [0.8, 0.2])
    ])

    transforms2 = TransformsPipeline([
        # NormalizeZScoreTransform(
        #     [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]
        # ),
    ])

    data = transforms(dataset)
    data2 = transforms2(dataset2)

    data_1, data_2 = [], []
    for name in data.get_names():
        for row in data[name]:
            data_1.append(row[0][0])

    for name in data2.get_names():
        for row in data2[name]:
            data_2.append(row[0][0])

    plot(data_1, data_2)
    # model = Sequential([
    #     Linear(4, 10, precision_bits=num_bits),
    #     ReLU(),
    #     Linear(10, 10, precision_bits=num_bits),
    #     ReLU(),
    #     Linear(10, 3, precision_bits=num_bits),
    #     SoftMax()
    # ])

    # output = pyrtl.Output(name='output')

    # output <<= model(a).to_wirevector()

    # print_stats("output")

    # sim = pyrtl.Simulation()

    # for d in data["test"]:
    #     a_vals = d[0]
    #     print(d)
    #     sim.step({
    #         'a_in': matrix.list_to_int(a_vals, n_bits=num_bits),
    #     })

    #     raw_matrix = matrix.matrix_wv_to_list(
    #         sim.inspect('output'), rows=1, columns=3, bits=16
    #     )
    #     print(raw_matrix)
    #     break
