#!/usr/bin/env python3
"""
vck190_first_smoke.py — ORBIT-G2 VCK190 First Board Smoke Test

Prerequisites:
  - VCK190 programmed with ORBIT-G2 Proto-B bitstream
  - PCIe card edge connected to host
  - Linux host with PCI device enumerated

Usage:
  python3 scripts/vck190_first_smoke.py [--bdf 0000:01:00.0]

Pass criteria:
  - lspci shows device
  - BAR0 size >= 1 MiB
  - BAR4 size >= 64 KiB
  - G2_ID read == 0x4732_0001
  - G2_VERSION, CAP0, TC0, IRQ all read with reset defaults
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmio_map import (
    G2_ID, G2_VERSION, G2_CAP0, G2_CAP1,
    BOOT_CAUSE, TC0_RUNSTATE, TC0_CTRL, TC0_FAULT_STATUS,
    IRQ_PENDING, IRQ_MASK, DMA_STATUS, DMA_ERR_CODE,
    TRACE_HEAD, TRACE_TAIL, TRACE_CTRL, OOM_STATE,
    PERF_FREEZE, MXU_BUSY_CYC_LO, MXU_TILE_COUNT,
)
from tools.orbit_mmap_backend import (
    LinuxPciBarOpener, MmapBackend,
    probe_orbit_device, discover_orbit_devices,
    DeviceNotFoundError, BarNotFoundError, BarSizeError,
    PermissionDeniedError, MmapOpenError,
)
from tools.orbit_device import OrbitDevice
from tools.orbit_dma import OrbitDma


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def run_smoke(bdf: str) -> bool:
    all_pass = True
    print(f"\n{'='*60}")
    print(f"ORBIT-G2 VCK190 First Smoke — BDF: {bdf}")
    print(f"{'='*60}\n")

    # ── Step 1: lspci ──────────────────────────────────────
    print("[1/7] PCIe Device Discovery")
    try:
        result = subprocess.run(
            ["lspci", "-d", "10ee:9038", "-nn"],
            capture_output=True, text=True, timeout=5
        )
        found = "10ee:9038" in result.stdout.lower() or bdf.split(":")[-1].split(".")[0] in result.stdout
        all_pass &= check("lspci device visible", len(result.stdout.strip()) > 0,
                          result.stdout.strip() or "NOT FOUND")
    except FileNotFoundError:
        check("lspci available", False, "lspci not installed")
        all_pass = False
    except Exception as e:
        check("lspci", False, str(e))
        all_pass = False

    # ── Step 2: Probe ──────────────────────────────────────
    print(f"\n[2/7] Device Probe")
    info = probe_orbit_device(bdf)
    all_pass &= check("Device found at BDF", info["found"])
    if not info["found"]:
        print(f"\n{'='*60}")
        print(f"SMOKE RESULT: {FAIL} (device not found)")
        print(f"{'='*60}")
        return False

    all_pass &= check("Vendor ID", info.get("vendor") == 0x10EE,
                       f"0x{info.get('vendor', 0):04x}")
    all_pass &= check("Device ID", info.get("device") == 0x9038,
                       f"0x{info.get('device', 0):04x}")

    bars = info.get("bars", {})
    for bidx, binfo in bars.items():
        all_pass &= check(f"BAR{bidx} size", binfo["ok"],
                          f"{binfo['size']:#x} (expected >= {binfo['expected']:#x})")

    # ── Step 3: Open BAR0 ─────────────────────────────────
    print(f"\n[3/7] BAR0 Open (MMIO Registers)")
    try:
        opener = LinuxPciBarOpener(bdf)
        backend = MmapBackend.from_opener(opener)
        check("BAR0 mmap", True)
    except DeviceNotFoundError as e:
        check("BAR0 mmap", False, f"DeviceNotFound: {e}")
        return False
    except BarNotFoundError as e:
        check("BAR0 mmap", False, f"BarNotFound: {e}")
        return False
    except BarSizeError as e:
        check("BAR0 mmap", False, f"BarSize: {e}")
        return False
    except PermissionDeniedError as e:
        check("BAR0 mmap", False, f"Permission: {e}")
        return False
    except MmapOpenError as e:
        check("BAR0 mmap", False, f"Mmap: {e}")
        return False

    # ── Step 4: Read G2_ID ────────────────────────────────
    print(f"\n[4/7] Register Reads (BAR0)")
    dev = OrbitDevice(backend)
    try:
        device_info = dev.connect()
        all_pass &= check("G2_ID", device_info.device_id == 0x4732_0001,
                          f"0x{device_info.device_id:08x}")
        all_pass &= check("G2_VERSION", device_info.version == 0x0001_0000,
                          f"0x{device_info.version:08x}")
        all_pass &= check("G2_CAP0", device_info.cap0 == 0x0000_0060,
                          f"0x{device_info.cap0:08x}")
    except Exception as e:
        check("G2_ID read", False, str(e))
        all_pass = False

    # ── Step 5: Status Reads ──────────────────────────────
    print(f"\n[5/7] Status Registers")
    try:
        tc = dev.read_tc_status()
        all_pass &= check("TC0_RUNSTATE = IDLE", tc.state == 0, f"state={tc.state}")
        all_pass &= check("TC0_CTRL.ENABLE", tc.enable, f"enable={tc.enable}")
        all_pass &= check("TC0_FAULT = 0", tc.fault_code == 0, f"fault=0x{tc.fault_code:02x}")

        irq = dev.poll_irq()
        all_pass &= check("IRQ_PENDING = 0", irq == 0, f"0x{irq:08x}")

        oom = dev.read_oom_state()
        all_pass &= check("OOM_STATE = NORMAL", oom == 0, f"state={oom}")
    except Exception as e:
        check("Status reads", False, str(e))
        all_pass = False

    # ── Step 6: BAR4 DMA Reads ────────────────────────────
    print(f"\n[6/7] DMA Registers (BAR4)")
    try:
        dma = OrbitDma(backend)
        st = dma.read_status()
        all_pass &= check("DMA_STATUS idle", not st.busy and not st.err,
                          f"busy={st.busy} done={st.done} err={st.err}")
        err = dma.read_error()
        all_pass &= check("DMA_ERR_CODE = 0", err == 0, f"0x{err:08x}")
    except Exception as e:
        check("DMA reads", False, str(e))
        all_pass = False

    # ── Step 7: Trace ─────────────────────────────────────
    print(f"\n[7/7] Trace Ring")
    try:
        dev.enable_trace()
        from tools.orbit_trace import read_trace_status
        ts = read_trace_status(backend)
        all_pass &= check("Trace head=0, tail=0", ts["head"] == 0 and ts["tail"] == 0,
                          f"head={ts['head']} tail={ts['tail']}")
    except Exception as e:
        check("Trace", False, str(e))
        all_pass = False

    # ── Cleanup ───────────────────────────────────────────
    try:
        opener.close()
    except Exception:
        pass

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_pass:
        print(f"SMOKE RESULT: {PASS}")
        print("VCK190 ORBIT-G2 Proto-B PCIe endpoint is ALIVE.")
    else:
        print(f"SMOKE RESULT: {FAIL}")
        print("See above for failing checks.")
    print(f"{'='*60}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="ORBIT-G2 VCK190 First Smoke")
    parser.add_argument("--bdf", default="0000:01:00.0",
                        help="PCI bus:device.function (default: 0000:01:00.0)")
    parser.add_argument("--discover", action="store_true",
                        help="Auto-discover ORBIT-G2 devices")
    args = parser.parse_args()

    if args.discover:
        devices = discover_orbit_devices()
        if not devices:
            print("No ORBIT-G2 devices found via sysfs scan.")
            return 1
        print(f"Found devices: {devices}")
        bdf = devices[0]
    else:
        bdf = args.bdf

    ok = run_smoke(bdf)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
