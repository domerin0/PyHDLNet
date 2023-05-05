import random


def linspace(start, stop, num_vals):
    delta = (stop-start)/(num_vals-1)
    return [start + i * delta for i in range(num_vals)]


def uniform(low, high, row_size, column_size=None):
    if column_size is None:
        return [random.uniform(low, high) for _ in range(row_size)]
    mat = []
    for c in range(row_size):
        mat.append([random.uniform(low, high) for _ in range(column_size)])
    return mat
