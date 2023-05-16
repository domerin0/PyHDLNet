import pyrtl
import pyrtl.rtllib.matrix as Matrix


def matrix_list_equal(result, expected_output, floored=False):
    """
    :param Matrix result: matrix that is the result of some operation we're testing
    :param list[list] expected_output: a list of lists to compare against
        the resulting matrix after simulation
    :param bool floored: needed to indicate that we're checking the result of
        a matrix subtraction, to ensure the matrix properly floored results to
        zero when needed (defaults to False)
    """
    output = pyrtl.Output(name='output')

    if isinstance(result, pyrtl.WireVector):
        output <<= result
    else:
        output <<= result.to_wirevector()

    sim = pyrtl.Simulation()
    sim.step({})

    if isinstance(result, pyrtl.WireVector):
        given_output = sim.inspect("output")
    else:
        given_output = Matrix.matrix_wv_to_list(
            sim.inspect("output"), result.rows, result.columns, result.bits
        )
    if isinstance(given_output, int):
        return given_output == expected_output
    else:
        for r in range(len(expected_output)):
            for c in range(len(expected_output[0])):
                expected = expected_output[r][c]
                if floored and expected < 0:
                    expected = 0
                if given_output[r][c] != expected:
                    print(given_output)
                    print(expected_output)
                    return False
    return True
