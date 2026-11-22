import os
from cocotb_test.simulator import run

# ACC_SPLIT: 누산기 캐리 체인 분할 (mac_pe.sv 참고). 둘 다 같은 결과를
# 내야 한다 — 분할은 Fmax 를 올리는 것이고 동작을 바꾸지 않는다.
ACC_SPLIT = int(os.environ.get("ACC_SPLIT", "0"))

run(
    simulator="icarus",
    toplevel="mac_array",
    module="test_mac_array",
    python_search=["sim/cocotb"],
    verilog_sources=[
        "rtl/mac_array.sv",
        "rtl/mac_pe.sv",
    ],
    parameters={"ACC_SPLIT": ACC_SPLIT},
    compile_args=["-g2012"],
    waves=True,
)
