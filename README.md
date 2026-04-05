# ORBIT-G2 — Custom LLM Inference Accelerator

> Custom SystemVerilog RTL + Python host stack for a dedicated LLM inference chip.  
> Full software-hardware closed loop verified: **237 tests** across RTL simulation, cocotb co-simulation, and Python host stack.

---

## What is this?

ORBIT-G2 is an open-source hardware accelerator for LLM inference. Built from scratch — RTL, host software, FPGA bitstream, everything.

- **23 RTL modules** in SystemVerilog (control plane + compute pipeline)
- **Python host stack** with HAL, CLI, descriptor packer, trace decoder, scheduler
- **Co-simulation**: Python host stack drives RTL via cocotb — GEMM end-to-end verified
- **FPGA target**: VCK190 (Versal VC1902) PCIe Gen4 x8
- **Bitstream generated**: Vivado 2025.2, synthesis + implementation + PDI complete

---

## Test Results

| Category | Tests | Status |
|----------|-------|--------|
| Python unit/integration | 208 | All pass |
| RTL cocotb (module-level) | 24 | All pass |
| Host-driven DUT (GEMM E2E) | 5 | All pass |
| **Total** | **237** | **All pass** |

### What the E2E test actually proves

The `test_host_gemm_e2e` test runs this full path on real RTL:

```
Python host stack
  → descriptor pack (CRC-8)
  → register staging (16 MMIO writes)
  → doorbell ring
  → desc_queue push
  → priority arbiter (Q3>Q0>Q1>Q2)
  → desc_fsm_v2 (CRC check + opcode validation)
  → gemm_top dispatch
  → gemm_core (DMA read act + wgt → MAC compute → DMA write result)
  → completion IRQ
  → trace ring event
  → host reads IRQ + trace + clears fault
```

All in one cocotb test. No mocks, no shortcuts — the Python host stack talks to the SystemVerilog DUT.

---

## Architecture

```
Host Software (Python)
  │
  ├── OrbitDevice HAL
  │     ├── connect / enqueue / poll / clear
  │     └── Backend abstraction
  │           ├── SimBackend (unit tests)
  │           ├── CocotbBackend (RTL simulation)
  │           └── MmapBackend (real hardware via PCIe BAR mmap)
  │
  ▼
RTL (SystemVerilog)
  │
  ├── g2_protob_top ─────────── Proto-B top (PCIe + DMA + control)
  │     ├── pcie_ep_versal ──── CPM PCIe Gen4 x8 adapter
  │     ├── dma_bridge ──────── DMA submit/status state machine
  │     └── g2_ctrl_top ─────── Proto-A control plane
  │           ├── reg_top ───── 48 MMIO registers (REG_SPEC)
  │           ├── desc_queue ── 4 descriptor queues + priority arbiter
  │           ├── desc_fsm_v2 ─ CRC / timeout / opcode validation
  │           ├── gemm_top ──── GEMM orchestrator
  │           │     ├── ctrl_fsm
  │           │     └── gemm_core (INT8 16×16 MAC array)
  │           ├── oom_guard ─── 4-state memory pressure controller
  │           ├── trace_ring ── 1K-entry debug event ring
  │           ├── irq_ctrl ──── 12-source interrupt controller (W1C)
  │           └── reset_seq ─── Reset sequencer (POR/SW/WDOG)
  │
  ▼
FPGA (VCK190 / VC1902)
  PCIe Gen4 x8 → BAR0 (1MB registers) + BAR4 (64KB DMA)
```

---

## Quick Start

### Run tests (no hardware needed)

```bash
pip install pytest
cd yua-t16
python -m pytest tests/ -v
# 208 passed
```

### Debug CLI

```bash
python -m tools.orbit_debug_protoa info
python -m tools.orbit_debug_protoa queue-status
python -m tools.orbit_debug_protoa tc-status
python -m tools.orbit_debug_protoa irq
python -m tools.orbit_debug_protoa trace-dump --count 16
python -m tools.orbit_debug_protoa doorbell --queue 0 --opcode 0x01
```

### RTL co-simulation (needs verilator + cocotb)

```bash
pip install cocotb verilator
# Host-driven GEMM E2E runs Python host stack against RTL DUT
# See tb/tb_g2_ctrl_top_host_e2e.py
```

### Build FPGA bitstream (needs Vivado 2025.2)

```bash
cd fpga/vck190
vivado -mode batch -source create_project.tcl
# Configure CPM endpoint in Vivado GUI, then synthesize
```

---

## Project Structure

```
yua-t16/
├── rtl/                       # 23 SystemVerilog modules
│   ├── g2_protob_top.sv       #   Proto-B top (PCIe + control)
│   ├── g2_ctrl_top.sv         #   Proto-A control plane
│   ├── pcie_ep_versal.sv      #   PCIe endpoint (CPM adapter)
│   ├── dma_bridge.sv          #   DMA state machine
│   ├── reg_top.sv             #   48-register MMIO bank
│   ├── desc_queue.sv          #   4-queue ring buffer
│   ├── desc_fsm_v2.sv         #   Descriptor validator
│   ├── gemm_top.sv            #   GEMM orchestrator
│   ├── gemm_core.sv           #   INT8 16x16 MAC + DMA
│   ├── oom_guard.sv           #   Memory pressure controller
│   ├── trace_ring.sv          #   Debug trace ring
│   ├── irq_ctrl.sv            #   Interrupt controller
│   ├── reset_seq.sv           #   Reset sequencer
│   ├── cdc_fifo.sv            #   Async FIFO
│   └── ...                    #   mac_array, mac_pe, kvc_core, etc.
│
├── tb/                        # 9 cocotb testbenches, 29 tests
│   ├── tb_g2_ctrl_top_host_e2e.py  # Host-driven GEMM E2E
│   ├── dma_responder.py            # DMA test memory model
│   └── ...
│
├── tools/                     # Python host stack (14 modules)
│   ├── orbit_device.py        #   Device HAL
│   ├── orbit_mmio_map.py      #   Register map SSOT
│   ├── orbit_desc.py          #   Descriptor packer + CRC
│   ├── orbit_scheduler.py     #   Op scheduler
│   ├── orbit_debug_protoa.py  #   Debug CLI
│   └── ...
│
├── tests/                     # 16 test files, 208 tests
├── fpga/vck190/               # Vivado project + Tcl scripts
├── scripts/                   # Board smoke test
└── docs/                      # 15 design documents
```

---

## Register Map

48 registers across 11 blocks. All addresses in [`tools/orbit_mmio_map.py`](tools/orbit_mmio_map.py).

| Block | Key Registers |
|-------|--------------|
| Global | G2_ID (`0x47320001`), VERSION, CAP0 |
| Reset | BOOT_CAUSE, SW_RESET, WDOG_CTRL |
| Queue | Q0-Q3 DOORBELL, STATUS, OVERFLOW (W1C) |
| DMA | SUBMIT_LO/HI, CTRL, STATUS, ERR_CODE |
| OOM | USAGE, RESERVED, STATE (NORMAL/PRESSURE/CRITICAL/EMERG) |
| TC0 | RUNSTATE (IDLE/FETCH/RUN/FAULT), CTRL, FAULT_STATUS |
| Perf | MXU_BUSY_CYCLES, TILE_COUNT, FREEZE |
| IRQ | PENDING (W1C, set-wins), MASK, FORCE, CAUSE_LAST |
| Trace | HEAD, TAIL, CTRL + 1K-entry read window |

---

## Status

| Milestone | Status |
|-----------|--------|
| RTL skeleton (7 new modules) | Done |
| Control plane integration | Done |
| MMIO device contract (48 registers) | Done |
| Python host stack + CLI | Done |
| Host-driven RTL co-simulation | Done — GEMM E2E proven |
| Proto-B PCIe/DMA contract | Done |
| Linux MMIO open path | Done |
| VCK190 Vivado project + CPM config | Done |
| **Bitstream (PDI) generated** | **Done** — 0 errors |
| Connect RTL to CPM Block Design | Next |
| VCK190 board bring-up | Needs board |

---

## License

MIT

---

*Built by YUA AI. Simulation-verified, bitstream-generated, awaiting silicon.*
