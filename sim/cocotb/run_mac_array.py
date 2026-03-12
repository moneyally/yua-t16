from cocotb_test.simulator import run

run(
    simulator="icarus",
    toplevel="mac_array",
    module="test_mac_array",
    python_search=["sim/cocotb"],
    verilog_sources=[
        "rtl/mac_array.sv",
        "rtl/mac_pe.sv",
    ],
    compile_args=["-g2012"],
    waves=True,
)
