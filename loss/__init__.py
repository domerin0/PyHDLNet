from typing import Any
from tensor import HDLTensor, HDLScalar
from abc import ABC, abstractmethod
from math import log


class LossFunction(ABC):
    @abstractmethod
    def forward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        pass


class CrossEntropyLoss(LossFunction):
    def __init__(
        self,
        batch_size: int,
        bits: int,
        signed: bool
    ) -> None:
        self.batch_size = HDLScalar(batch_size)
        self.loss_accumulator = HDLTensor(
            1, 1, bits=bits*2,
            signed=signed,
            value=[[0]]
        )

    def forward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        '''
        Inputs:
            y_pred: HDLTensor (batch size, output_dim)
            y_true: HDLTensor (1, output_dim)
        Returns: 
            HDLTensor of the loss averaged over the batch 
        '''
        for i in range(y_pred.rows):
            for j in range(y_pred.columns):
                self.loss_accumulator[0, 0] = self.loss_accumulator[0, 0] + \
                    y_true[0, j] * y_pred.log2(i, j)

            # divide accumulated loss by batch size
            self.loss_accumulator[0, 0] = self.loss_accumulator.divide_scalar(
                self.batch_size.value,
                row=0, col=0
            )
        # return the loss
        return self.loss_accumulator

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class MeanSquaredErrorLoss:
    def __init__(self, batch_size, output_dim) -> None:
        self.accumulator = 0

    def forward(y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        raise NotImplementedError(
            "MeanSquaredErrorLoss.forward() not implemented")
