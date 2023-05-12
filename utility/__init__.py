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


def uniform_int(max_int, min_int, row_size, column_size=None):
    if column_size is None:
        return [random.randint(min_int, max_int) for _ in range(row_size)]
    mat = []
    for c in range(row_size):
        mat.append([random.randint(min_int, max_int)
                    for _ in range(column_size)])
    return mat


def clamp(x, u, l):
    if x > u:
        return u
    elif x < l:
        return l
    else:
        return x


def quantize(x, scale_factor, zero_point, max_int, min_int):
    return clamp(round(scale_factor * x - zero_point), max_int, min_int)
