from typing import Any
from tensor import HDLTensor, HDLScalar
from abc import ABC, abstractmethod
from math import log


class LossFunction(ABC):
    @abstractmethod
    def forward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        pass

    @abstractmethod
    def backward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        pass


class CrossEntropyLoss(LossFunction):
    def __init__(
        self,
    ) -> None:
        self.batch_size = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        '''
        Inputs:
            y_pred: HDLTensor (batch size, output_dim)
            y_true: HDLTensor (1, output_dim)
        Returns: 
            HDLTensor of the loss averaged over the batch 
        '''
        if self.batch_size is None:
            self.batch_size = HDLScalar(y_pred.rows)
            self.loss_accumulator = HDLTensor(
                1, 1, bits=y_pred.bits*2,
                signed=y_pred.signed,
                value=[[0]]
            )
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

    def backward(self, y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        """
        Computes the gradinets of the loss_fn wrt to the predicted labels
        Args:
            y_pred : predicted labels
            y_true : true labels
        Returns:
         da : derivative of loss_fn wrt to the predicted labels
        """
        # derivative of loss_fn with respect to a [predicted labels]
        if self.ones is None:
            max_val = 2**y_true.bits if not y_true.signed else 2**(
                y_true.bits-1)
            rows = y_true.rows
            columns = y_true.columns
            self.ones = HDLTensor(
                rows,
                columns,
                bits=y_true.bits,
                value=[
                    [max_val for _ in range(columns)]
                    for _ in range(rows)
                ]
            )
        da = - ((y_true // y_pred) - ((self.ones - y_true) //
                (self.ones - y_pred)))
        return da

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class MeanSquaredErrorLoss:
    def __init__(self, batch_size, output_dim) -> None:
        self.accumulator = 0

    def forward(y_pred: HDLTensor, y_true: HDLTensor) -> HDLTensor:
        raise NotImplementedError(
            "MeanSquaredErrorLoss.forward() not implemented")
