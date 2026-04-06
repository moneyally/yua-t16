"""
tb_g3_int_multistep_loop.py — G3-INT-003: Multi-Step Training Loop

Verifies param/m/v/scale state accumulates across training steps.
Loop: backward → optimizer → loss_scaler, repeated N times.
Active region: 16×16. Forward omitted (update recurrence proof).
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct, math

DIM = 16

def f2fp32(f): return struct.unpack('>I', struct.pack('>f', f))[0]
def fp32f(b): return struct.unpack('>f', struct.pack('>I', b & 0xFFFFFFFF))[0]
def f2bf16(f): return (struct.unpack('>I', struct.pack('>f', f))[0] >> 16) & 0xFFFF
def set_bf16_id(m, s=1.0):
    for i in range(DIM):
        for j in range(DIM): m[i][j].value = f2bf16(s if i==j else 0.0)
def set_fp32_diag(m, v):
    for i in range(DIM):
        for j in range(DIM): m[i][j].value = f2fp32(v if i==j else 0.0)
def set_fp32_z(m):
    for i in range(DIM):
        for j in range(DIM): m[i][j].value = 0

def py_adam(p,g,m,v,lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
    mn=b1*m+(1-b1)*g; vn=b2*v+(1-b2)*g*g
    return p-lr*mn/(math.sqrt(max(vn,0))+eps), mn, vn

async def reset_dut(dut):
    dut.rst_n.value=0; dut.start.value=0; dut.num_steps.value=2
    dut.adamw_enable.value=0; dut.overflow_inject.value=0
    dut.lr_fp32.value=f2fp32(0.001)
    dut.beta1_fp32.value=f2fp32(0.9)
    dut.beta2_fp32.value=f2fp32(0.999)
    dut.epsilon_fp32.value=f2fp32(1e-8)
    dut.weight_decay_fp32.value=f2fp32(0.0)
    dut.init_scale_fp32.value=f2fp32(32768.0)
    dut.growth_factor_fp32.value=f2fp32(2.0)
    dut.backoff_factor_fp32.value=f2fp32(0.5)
    dut.growth_interval.value=2
    dut.min_scale_fp32.value=f2fp32(1.0)
    dut.max_scale_fp32.value=f2fp32(1e7)
    set_bf16_id(dut.x_bf16, 1.0)
    set_bf16_id(dut.w_bf16, 1.0)
    set_bf16_id(dut.dy_bf16, 0.5)
    set_fp32_diag(dut.param_init, 5.0)
    set_fp32_z(dut.m_init)
    set_fp32_z(dut.v_init)
    await Timer(100, unit="ns"); dut.rst_n.value=1
    for _ in range(5): await RisingEdge(dut.clk)

async def run_loop(dut, w=3000):
    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0
    for _ in range(w):
        await RisingEdge(dut.clk)
        if dut.loop_done_pulse.value==1: return True
    return False


@cocotb.test()
async def test_two_step_param_accumulates(dut):
    """2 steps: param changes after each. Step2 result ≠ step1 result."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.num_steps.value = 2

    # Python reference: 2 Adam steps
    # dW = X^T * dY = I * 0.5*I = 0.5*I (diagonal)
    g = 0.5
    p0, m0, v0 = 5.0, 0.0, 0.0
    p1, m1, v1 = py_adam(p0, g, m0, v0)
    p2, m2, v2 = py_adam(p1, g, m1, v1)

    ok = await run_loop(dut)
    assert ok, "loop_done not seen"
    assert int(dut.err_code.value)==0

    p_out = fp32f(int(dut.cur_param[0][0].value))
    dut._log.info(f"After 2 steps: param={p_out}, ref_step1={p1}, ref_step2={p2}")
    assert abs(p_out - p2) < abs(p2)*0.02 + 1e-5, f"param after 2 steps: {p_out} vs {p2}"
    assert abs(p_out - p1) > 1e-6, "Step2 should differ from step1"


@cocotb.test()
async def test_m_v_state_accumulates(dut):
    """m and v should be nonzero and growing after 2 steps."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.num_steps.value = 2

    g = 0.5
    _, m1, v1 = py_adam(5.0, g, 0.0, 0.0)
    _, m2, v2 = py_adam(0, g, m1, v1)  # param doesn't matter for m/v

    ok = await run_loop(dut)
    assert ok

    m_out = fp32f(int(dut.cur_m[0][0].value))
    v_out = fp32f(int(dut.cur_v[0][0].value))
    dut._log.info(f"m={m_out} (ref={m2}), v={v_out} (ref={v2})")
    assert abs(m_out - m2) < abs(m2)*0.02 + 1e-6
    assert abs(v_out - v2) < abs(v2)*0.02 + 1e-6
    assert m_out != 0.0, "m should be nonzero"
    assert v_out != 0.0, "v should be nonzero"


@cocotb.test()
async def test_loss_scale_grows_after_success(dut):
    """2 success steps with growth_interval=2 → scale doubles."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.num_steps.value = 2
    dut.growth_interval.value = 2
    dut.overflow_inject.value = 0

    ok = await run_loop(dut)
    assert ok

    s = fp32f(int(dut.cur_scale_fp32.value))
    dut._log.info(f"Scale after 2 success steps: {s}")
    assert abs(s - 65536.0) < 1.0, f"Expected 32768*2=65536, got {s}"


@cocotb.test()
async def test_loss_scale_drops_on_overflow(dut):
    """Step1 success, step2 overflow → scale drops."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.num_steps.value = 3
    dut.growth_interval.value = 10  # won't reach
    dut.overflow_inject.value = 0

    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0

    # Wait for step1 done (no overflow)
    for _ in range(1500):
        await RisingEdge(dut.clk)
        if dut.step_done_pulse.value==1: break

    # Inject overflow for step2
    dut.overflow_inject.value = 1
    for _ in range(1500):
        await RisingEdge(dut.clk)
        if dut.step_done_pulse.value==1: break
    dut.overflow_inject.value = 0

    # Wait for loop done
    for _ in range(1500):
        await RisingEdge(dut.clk)
        if dut.loop_done_pulse.value==1: break

    s = fp32f(int(dut.cur_scale_fp32.value))
    dut._log.info(f"Scale after overflow: {s}")
    # After step1 success + step2 overflow: 32768 → 16384. Step3 success: no change (cnt=1<10)
    assert s < 32768.0, f"Scale should have decreased, got {s}"


@cocotb.test()
async def test_reset_mid_loop(dut):
    """Reset during loop returns to IDLE."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.num_steps.value = 5
    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0
    for _ in range(30): await RisingEdge(dut.clk)
    dut.rst_n.value=0
    for _ in range(2): await RisingEdge(dut.clk)
    dut.rst_n.value=1
    for _ in range(5): await RisingEdge(dut.clk)
    assert dut.busy.value==0


@cocotb.test()
async def test_busy_step_done_loop_done(dut):
    """busy high, step_done pulses per step, loop_done at end."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.num_steps.value = 2
    dut.start.value=1; await RisingEdge(dut.clk); dut.start.value=0

    saw_busy=False; step_dones=0
    for _ in range(3000):
        await RisingEdge(dut.clk)
        if dut.busy.value==1: saw_busy=True
        if dut.step_done_pulse.value==1: step_dones+=1
        if dut.loop_done_pulse.value==1: break

    assert saw_busy
    assert step_dones==2, f"Expected 2 step_dones, got {step_dones}"
    assert dut.loop_done_pulse.value==1
