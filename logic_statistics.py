import pyrtl


def print_stats(name):
    ta = pyrtl.TimingAnalysis()
    print(f"Max frequency: {ta.max_freq()} MhZ")
    print(f"Max timing delay: {ta.max_length()} ps")
    print(
        f"Logic count: {len({l for l in pyrtl.working_block().logic if l.op not in 'wcs'})}")
    logic, _ = pyrtl.area_estimation()
    print(f"Logic area est: {logic}")


def write_verilog(name):
    with open(f"{name}.v", 'w') as f:
        pyrtl.output_to_verilog(f)


def write_graphviz(name):
    with open(f"{name}.gv", 'w') as f:
        pyrtl.output_to_graphviz(f)
