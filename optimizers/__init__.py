from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self) -> None:
        pass


class AutoGrad:
    def __init__(self) -> None:
        pass

    def backward(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def __repr__(self) -> str:
        return "AutoGrad()"

    def __str__(self) -> str:
        return "AutoGrad()"