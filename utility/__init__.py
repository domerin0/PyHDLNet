import random
from typing import List, Any


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


def uniform_int(max_int, min_int, row_size, column_size=None):
    if column_size is None:
        return [random.randint(min_int, max_int) for _ in range(row_size)]
    mat = []
    for c in range(row_size):
        mat.append([random.randint(min_int, max_int)
                    for _ in range(column_size)])
    return mat


def reduce_tensors(a, b, func=None):
    if func is None:
        def func(x, y): return x + y
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError(
                "Tensors must be the same size, got {0} and {1}"
                .format(len(a), len(b)
                        ))
        if isinstance(a[0], list):
            for i in range(len(a)):
                b[i] = reduce_tensors(a[i], b[i], func)
        else:
            for i in range(len(a)):
                b[i] = func(a[i], b[i])
    else:
        b = func(a, b)
    return b


def apply_tensor(tensor, func):
    if isinstance(tensor, list):
        if isinstance(tensor[0], list):
            for i in range(len(tensor)):
                tensor[i] = apply_tensor(tensor[i], func)
        else:
            for i in range(len(tensor)):
                tensor[i] = func(tensor[i])
    else:
        tensor = func(tensor)
    return tensor


def apply_tensor_per_channel(tensor, func):
    if isinstance(tensor, list):
        if isinstance(tensor[0], list):
            for i in range(len(tensor)):
                tensor[i] = apply_tensor(tensor[i], func)
        else:
            for i in range(len(tensor)):
                tensor[i] = func(i, tensor[i])
    else:
        tensor = func(0, tensor)
    return tensor


def clamp(x, u, l):
    if x > u:
        return u
    elif x < l:
        return l
    else:
        return x


def max_tensor(tensor, channel=None):
    max_val = None
    if isinstance(tensor, list):
        if isinstance(tensor[0], list):
            for i in range(len(tensor)):
                calc = max_tensor(tensor[i])
                max_val = calc if max_val is None else max(max_val, calc)
        else:
            if channel is None:
                for i in range(len(tensor)):
                    if max_val is None or tensor[i] > max_val:
                        max_val = tensor[i]
            else:
                if max_val is None or tensor[channel] > max_val:
                    max_val = tensor[channel]
    else:
        max_val = tensor if max_val is None else max(max_val, tensor)
    return max_val


def min_tensor(tensor, channel=None):
    min_val = None
    if isinstance(tensor, list):
        if isinstance(tensor[0], list):
            for i in range(len(tensor)):
                calc = min_tensor(tensor[i])
                min_val = calc if min_val is None else min(min_val, calc)
        else:
            if channel is None:
                for i in range(len(tensor)):
                    if min_val is None or tensor[i] < min_val:
                        min_val = tensor[i]
            else:
                if min_val is None or tensor[channel] < min_val:
                    min_val = tensor[channel]
    else:
        min_val = tensor if min_val is None else min(min_val, tensor)
    return min_val
