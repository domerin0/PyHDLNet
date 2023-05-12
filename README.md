# PyHDLNet
Convenient nn library that exports itself to HDL code for custom FPGA-based accelerators.

This is a work in progress, as such the docs will be ready when the library is ready for a beta release, and more ops are added.

## Done:

- Linear
- ReLU
- Bit Truncation
- SoftMax

## Currently In progress:

Building infrastructure for training.

## TODO:

- Convolution
- Multi-head Attention
- Sigmoid
- Tanh
- BatchNorm
- Dropout
- MaxPool
- AveragePool
- Flatten
- Reshape
- Concat
- Embedding
- LSTM/GRU








### Test

```bash
python -m unittest discover -s test
```