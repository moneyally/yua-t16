"""
tb_g3_int_2chip_fabric.py — G3-INT-005: Fabric-Connected 2-Chip Training

Gradient travels through fabric send/recv instead of direct peer feed.
SUM reduction. 16×16 active region.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct, math

DIM = 16

def float_to_bf16(f):
    return (struct.unpack('>I', struct.pack('>f', f))[0] >> 16) & 0xFFFF
def float_to_fp32(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]
def fp32_to_float(b):
    return struct.unpack('>f', struct.pack('>I', b & 0xFFFFFFFF))[0]
def set_bf16_id(m, s=1.0):
    for i in range(DIM):
        for j in range(DIM):
            m[i][j].value = float_to_bf16(s if i==j else 0.0)
def set_bf16_z(m):
    for i in range(DIM):
        for j in range(DIM): m[i][j].value = 0
def set_fp32_diag(m, v):
    for i in range(DIM):
        for j in range(DIM): m[i][j].value = float_to_fp32(v if i==j else 0.0)
def set_fp32_z(m):
    for i in range(DIM):
        for j in range(DIM): m[i][j].value = 0

async def reset_dut(dut):
    dut.rst_n.value=0; dut.start.value=0; dut.adamw_enable.value=0
    dut.lr_fp32.value=float_to_fp32(0.001)
    dut.beta1_fp32.value=float_to_fp32(0.9)
    dut.beta2_fp32.value=float_to_fp32(0.999)
    dut.epsilon_fp32.value=float_to_fp32(1e-8)
    dut.weight_decay_fp32.value=float_to_fp32(0.0)
    for m in [dut.x0_bf16,dut.w0_bf16,dut.dy0_bf16,dut.x1_bf16,dut.w1_bf16,dut.dy1_bf16]:
        set_bf16_z(m)
    for m in [dut.param0_in,dut.m0_in,dut.v0_in,dut.param1_in,dut.m1_in,dut.v1_in]:
        set_fp32_z(m)
    await Timer(100, unit="ns"); dut.rst_n.value=1
    for _ in range(5): await RisingEdge(dut.clk)

async def run(dut, w=5000):
    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0
    for _ in range(w):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value==1: return True
    return False

def py_adam(p,g,m,v,lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
    mn=b1*m+(1-b1)*g; vn=b2*v+(1-b2)*g*g
    return p-lr*mn/(math.sqrt(max(vn,0))+eps),mn,vn

@cocotb.test()
async def test_2chip_fabric_symmetric(dut):
    """Same inputs both chips → param0==param1 via fabric path."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    for m in [dut.x0_bf16,dut.x1_bf16]: set_bf16_id(m,1.0)
    for m in [dut.w0_bf16,dut.w1_bf16]: set_bf16_id(m,1.0)
    for m in [dut.dy0_bf16,dut.dy1_bf16]: set_bf16_id(m,0.5)
    for m in [dut.param0_in,dut.param1_in]: set_fp32_diag(m,3.0)
    for m in [dut.m0_in,dut.m1_in,dut.v0_in,dut.v1_in]: set_fp32_z(m)

    ok=await run(dut)
    assert ok, "done not seen"
    assert int(dut.err_code.value)==0

    p0=fp32_to_float(int(dut.param0_out[0][0].value))
    p1=fp32_to_float(int(dut.param1_out[0][0].value))
    dut._log.info(f"fabric symmetric: p0={p0}, p1={p1}")
    assert abs(p0-p1)<1e-5, f"Symmetric should match: p0={p0}, p1={p1}"

@cocotb.test()
async def test_2chip_fabric_asymmetric(dut):
    """dY0=0.3, dY1=0.7 → reduced=1.0 → same Adam update."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    for m in [dut.x0_bf16,dut.x1_bf16]: set_bf16_id(m,1.0)
    for m in [dut.w0_bf16,dut.w1_bf16]: set_bf16_id(m,1.0)
    set_bf16_id(dut.dy0_bf16,0.3); set_bf16_id(dut.dy1_bf16,0.7)
    for m in [dut.param0_in,dut.param1_in]: set_fp32_diag(m,5.0)
    for m in [dut.m0_in,dut.m1_in,dut.v0_in,dut.v1_in]: set_fp32_z(m)

    ok=await run(dut)
    assert ok
    assert int(dut.err_code.value)==0

    rdw0=fp32_to_float(int(dut.reduced_dw0[0][0].value))
    rdw1=fp32_to_float(int(dut.reduced_dw1[0][0].value))
    dut._log.info(f"reduced: dw0={rdw0}, dw1={rdw1}")
    assert abs(rdw0-1.0)<0.1
    assert abs(rdw0-rdw1)<0.01, "Both chips should get same reduced_dw"

    p0=fp32_to_float(int(dut.param0_out[0][0].value))
    p1=fp32_to_float(int(dut.param1_out[0][0].value))
    p_ref,_,_=py_adam(5.0,rdw0,0,0)
    dut._log.info(f"fabric async: p0={p0}, p1={p1}, ref={p_ref}")
    assert abs(p0-p_ref)<abs(p_ref)*0.02+1e-5
    assert abs(p0-p1)<1e-5

@cocotb.test()
async def test_busy_done(dut):
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    for m in [dut.x0_bf16,dut.x1_bf16]: set_bf16_id(m)
    for m in [dut.dy0_bf16,dut.dy1_bf16]: set_bf16_id(m)
    for m in [dut.param0_in,dut.param1_in]: set_fp32_diag(m,1.0)
    for m in [dut.m0_in,dut.m1_in,dut.v0_in,dut.v1_in]: set_fp32_z(m)
    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0
    saw=False
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if dut.busy.value==1: saw=True
        if dut.done_pulse.value==1: break
    assert saw and dut.done_pulse.value==1

@cocotb.test()
async def test_reset_mid(dut):
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    for m in [dut.x0_bf16,dut.x1_bf16]: set_bf16_id(m)
    for m in [dut.dy0_bf16,dut.dy1_bf16]: set_bf16_id(m)
    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0
    for _ in range(30): await RisingEdge(dut.clk)
    dut.rst_n.value=0
    for _ in range(2): await RisingEdge(dut.clk)
    dut.rst_n.value=1
    for _ in range(3): await RisingEdge(dut.clk)
    assert dut.busy.value==0
