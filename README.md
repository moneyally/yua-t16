# YUA-T16 / ORBIT-G1 — Open-Source LLM Inference Accelerator RTL

> **Open-source SystemVerilog RTL for a custom LLM inference accelerator chip.**
> Designed to run GPT-class MoE Transformer models (e.g., GPT-OSS-20B) end-to-end on dedicated hardware.

---

## ⚠️ Verification Status

| Component | Simulation | Real Hardware |
|-----------|-----------|---------------|
| YUA-T16 GEMM tile (INT8) | ✅ cocotb PASS — bit-exact vs NumPy | ❌ Not verified |
| VPU (RMSNorm / SiLU / RoPE / Softmax) | ✅ cocotb 8/8 PASS | ❌ Not verified |
| KVC Controller (KV-Cache) | ✅ cocotb 4/4 PASS | ❌ Not verified |
| MoE Router (top-k) | ✅ cocotb 3/3 PASS | ❌ Not verified |
| INT4 GEMM (AWQ dequant) | ✅ cocotb 3/3 PASS | ❌ Not verified |
| Full Transformer forward pass | ✅ integration test PASS | ❌ Not verified |

**Simulation verification is complete. Real hardware verification (FPGA timing, power, memory bandwidth, thermal stability) has not been performed.**
FPGA synthesis and board-level validation are the critical next step — contributions welcome.

---

## What Is This

**ORBIT-G1** is a PCIe accelerator architecture designed to execute large language model inference entirely in hardware. It uses an array of **YUA-T16** GEMM tiles plus purpose-built units for every operation in the Transformer forward pass.

Target model: **GPT-OSS-20B** (MoE Transformer, 32 experts, Apache 2.0)
Target platform: Arty A7-100T FPGA (prototype) → ASIC (long-term)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ORBIT-G1 v2                       │
│                                                      │
│  PCIe Gen4 x16                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │           Command Processor                  │    │
│  │   Descriptor Queue × 4                      │    │
│  └──────┬───────────┬────────────┬─────────────┘    │
│         │           │            │                   │
│  ┌──────▼──┐  ┌─────▼─────┐  ┌──▼──────────────┐   │
│  │Compute  │  │    VPU    │  │  KVC + MoE      │   │
│  │Clusters │  │(RMSNorm,  │  │  Controller     │   │
│  │         │  │SiLU,RoPE, │  │                 │   │
│  │N×YUA-T16│  │Softmax,   │  │ KV-Cache GDDR6  │   │
│  │tiles    │  │Residual)  │  │ MoE Router      │   │
│  │INT8/INT4│  │256-wide   │  │ top-k select    │   │
│  └──────┬──┘  └─────┬─────┘  └──┬──────────────┘   │
│         └───────────┴───────────┘                   │
│                     │                               │
│         ┌───────────▼──────────────┐                │
│         │     Global Memory        │                │
│         │  GDDR6 (16GB or 32GB)   │                │
│         │  Weights + KV-Cache      │                │
│         └──────────────────────────┘                │
└─────────────────────────────────────────────────────┘
```

### Inference execution: 21-step descriptor sequence

One token decode pass through a single Transformer layer:

```
① DMA_2D       — load token embedding
② GEMM_INT4    — QKV projection
③ VECTOR_OP    — RoPE on Q, K
④ KVC_WRITE    — store new K, V to cache
⑤ KVC_READ     — fetch full sequence K, V
⑥ GEMM_INT4    — Q @ K^T (attention scores)
⑦ VECTOR_OP    — scale(1/√d) + softmax
⑧ GEMM_INT4    — scores @ V
⑨ GEMM_INT4    — output projection
⑩ VECTOR_OP    — residual add
⑪ VECTOR_OP    — RMSNorm
⑫ GEMM_INT4    — MoE router logits
⑬ MOE_ROUTE    — top-2 expert selection
⑭ GEMM_INT4    — gate_proj (per expert)
⑮ VECTOR_OP    — SiLU(gate) * up
⑯ GEMM_INT4    — down_proj (per expert)
⑰ VECTOR_OP    — expert weighted sum + residual
⑱ BARRIER
⑲ GEMM_INT4    — LM head projection
⑳ VECTOR_OP    — softmax → argmax
㉑ EVENT        — emit next token to host
```

All 21 steps verified end-to-end in `sim/integration/test_llm_forward.py`.

---

## Repository Structure

```
yua-t16/
├── rtl/
│   ├── mac_pe.sv           — Single INT8×INT8→INT32 MAC PE
│   ├── mac_array.sv        — 16×16 MAC PE array
│   ├── gemm_core.sv        — GEMM compute FSM + SRAM control
│   ├── gemm_top.sv         — Top-level wrapper (descriptor interface)
│   ├── ctrl_fsm.sv         — 64-byte descriptor decode FSM
│   ├── act_sram.sv         — Activation SRAM
│   ├── wgt_sram.sv         — Weight SRAM
│   ├── vpu_core.sv         — 256-wide SIMD VPU (RMSNorm/SiLU/RoPE/Softmax/CLAMP)
│   ├── kvc_core.sv         — KV-Cache controller (PagedAttention-style)
│   ├── moe_router.sv       — MoE top-k router (numerically stable softmax)
│   └── gemm_int4.sv        — INT4 weight GEMM with AWQ FP16 dequantization
├── sim/
│   ├── cocotb/             — cocotb testbenches (Icarus Verilog 12)
│   │   ├── test_gemm_top.py
│   │   ├── test_mac_array.py
│   │   ├── test_vpu.py
│   │   ├── test_kvc.py
│   │   ├── test_moe.py
│   │   └── test_gemm_int4.py
│   ├── integration/
│   │   └── test_llm_forward.py  — Full Transformer layer forward pass
│   └── golden/             — NumPy reference models
└── spec/
    ├── yua-t16.md          — YUA-T16 GEMM tile specification
    ├── yua-t16-v2.md       — YUA-T16 v2 (INT4 extension)
    ├── orbit-g1.md         — ORBIT-G1 system architecture
    ├── descriptor.md       — Descriptor format specification
    ├── vpu.md              — VPU design specification
    ├── kvc.md              — KV-Cache controller specification
    └── yua-llm-hw-design.md — Full LLM hardware design doc
```

---

## Running the Simulations

### Prerequisites

```bash
# Icarus Verilog (simulator)
sudo apt install iverilog        # Ubuntu/Debian
brew install icarus-verilog      # macOS

# Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install cocotb cocotb-test pytest numpy
```

### Run individual component tests

```bash
cd /path/to/yua-t16
source .venv/bin/activate

python sim/cocotb/run_gemm_top.py    # GEMM tile
python sim/cocotb/run_vpu.py         # VPU (8 tests)
python sim/cocotb/run_kvc.py         # KV-Cache controller
python sim/cocotb/run_moe.py         # MoE router
python sim/cocotb/run_gemm_int4.py   # INT4 GEMM
```

### Run integration test

```bash
python sim/integration/test_llm_forward.py
```

Expected output:
```
[PASS] test_qkv_attention    — Q/K/V projection + RoPE + attention + output projection
[PASS] test_moe_ffn          — RMSNorm + MoE routing + expert FFN + residual
[PASS] test_full_forward     — Full 21-step descriptor sequence, next_token=204
```

---

## Design Decisions

### Icarus Verilog 12 compatibility
All behavioral RTL avoids known Icarus 12 bugs:
- No dynamic bit-selects inside `always` blocks — use `generate/assign` outside
- No `real` variables with non-blocking assignments in `always_ff`
- All behavioral models use `always @(posedge clk)` with blocking assignments

### FP16 arithmetic
VPU and INT4 GEMM use IEEE 754 FP16 throughout, implemented in software-style `real` arithmetic for behavioral simulation. Synthesizable RTL (Q8.8 fixed-point) is in `rtl/vpu_core_synth.sv` (WIP).

### Descriptor-driven execution
All compute is initiated via 64-byte descriptors submitted to a command queue. This decouples the software scheduler from hardware execution and enables pipelining across descriptor types.

---

## What Needs Hardware Verification

The following have **not** been measured on real hardware:

- **Timing closure** — whether 150 MHz target is achievable on Arty A7-100T
- **Resource utilization** — LUT/DSP/BRAM fit within xc7a100t
- **Power consumption** — estimated, not measured
- **Memory bandwidth** — GDDR6 throughput vs actual token/s
- **Long-term stability** — thermal behavior, error rate under sustained load

**If you have an FPGA board (Arty A7, Nexys A7, or similar Xilinx 7-series), FPGA synthesis contributions are the highest priority need.**

---

## Roadmap

```
[DONE] Phase A — Behavioral RTL + simulation
  ✅ YUA-T16 GEMM tile
  ✅ VPU (all elementwise ops)
  ✅ KVC Controller
  ✅ MoE Router
  ✅ INT4 GEMM (AWQ)
  ✅ Full LLM forward pass integration test

[IN PROGRESS] Phase B — Synthesizable RTL
  🔄 vpu_core_synth.sv (Q8.8 fixed-point, no real/exp/sqrt)
  ⏳ FPGA synthesis report (Vivado, xc7a100t)
  ⏳ Timing closure at 150 MHz

[TODO] Phase C — FPGA board validation
  ⏳ Arty A7-100T bitstream
  ⏳ PCIe DMA test
  ⏳ Token throughput measurement

[TODO] Phase D — Software stack
  ⏳ Linux kernel PCIe driver
  ⏳ Userspace runtime library (C++)
  ⏳ OpenAI-compatible inference server
```

---

## Contributing

Highest-priority contributions:

1. **FPGA synthesis** — port behavioral RTL to synthesizable, run Vivado, report utilization/timing
2. **PCIe driver skeleton** — Linux kernel driver for descriptor queue submission
3. **Softmax/exp LUT** — fixed-point exp() approximation for VPU synthesis
4. **Bug reports** — if simulation tests fail on your setup, open an issue with Icarus version + OS

---

## Specification Documents

Full design documentation in `spec/`:

- [`spec/orbit-g1.md`](spec/orbit-g1.md) — System architecture
- [`spec/yua-llm-hw-design.md`](spec/yua-llm-hw-design.md) — LLM inference hardware design (GPT-OSS-20B mapping)
- [`spec/descriptor.md`](spec/descriptor.md) — Descriptor format v1/v2
- [`spec/vpu.md`](spec/vpu.md) — VPU design
- [`spec/kvc.md`](spec/kvc.md) — KV-Cache controller design

---

## License

MIT

---

*ORBIT-G1 is an independent open-source hardware project. Not affiliated with any commercial chip vendor.*
*Simulation-verified. Real hardware verification pending.*
