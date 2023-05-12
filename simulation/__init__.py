import pyrtl
import pyrtl.rtllib.matrix as matrix
from nn import NeuralModule
from dataloaders import Dataset
from abc import ABC, abstractmethod


class SimulationHarness(ABC):
    @abstractmethod
    def run(self, model: NeuralModule, data: Dataset) -> None:
        pass


class InferenceSimulationHarness(SimulationHarness):
    def __init__(self, model, warmup_tensor) -> None:
        # pyrtl.synthesize()
        # pyrtl.optimize()
        outputs = model(warmup_tensor)
        self.model = model
        self.vec_outputs = outputs
        if not isinstance(outputs, list):
            self.vec_outputs = [outputs]

    def run(self, data: Dataset, bits: int) -> None:
        self.model.simulate()

        raw_matrices = []
        outputs = []
        for i in range(len(self.vec_outputs)):
            output = pyrtl.Output(name="output{0}".format(i))
            output <<= self.vec_outputs[i].to_wirevector()
            outputs.append(output)

        sim_trace = pyrtl.SimulationTrace()
        sim = pyrtl.FastSimulation(tracer=sim_trace)

        for key in list(data.get_names()):
            for row in data[key]:
                sim.step({
                    'net_input': matrix.list_to_int([row[0]], n_bits=bits),
                })
                for i in range(len(outputs)):
                    opt = matrix.matrix_wv_to_list(
                        sim.inspect('output{0}'.format(i)),
                        rows=self.vec_outputs[i].rows,
                        columns=self.vec_outputs[i].columns,
                        bits=self.vec_outputs[i].bits
                    )
                    raw_matrices.append(opt)
        return raw_matrices
