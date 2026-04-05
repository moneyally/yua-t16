"""test_protoa_debug_flow.py — End-to-end host debug flow with SimBackend.

Simulates: reset -> info -> queue submit -> tc-status -> trace -> irq/fault clear
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_backend import SimBackend
from tools.orbit_mmio_map import (
    G2_ID, G2_VERSION, BOOT_CAUSE, BOOT_CAUSE_POR,
    TC0_RUNSTATE, TC0_CTRL, TC0_FAULT_STATUS,
    IRQ_PENDING, IRQ_MASK, IrqBit,
    TRACE_HEAD, TRACE_TAIL, TRACE_CTRL, TRACE_WIN_BASE, TRACE_META_BASE,
    OOM_USAGE_LO, OOM_STATE,
    Q0_STATUS, Q_OVERFLOW,
    BASE,
)
from tools.orbit_desc import pack_nop, stage_and_doorbell
from tools.orbit_poll import (
    read_irq_status, clear_all_irq, read_boot_cause,
    read_oom_status, clear_fault,
)
from tools.orbit_trace import read_trace_status
from tools.orbit_debug_protoa import cmd_info, cmd_queue_status, cmd_tc_status


class TestProtoADebugFlow:
    def setup_method(self):
        self.b = SimBackend()
        # Simulate POR boot cause
        self.b.set_ro(BOOT_CAUSE.offset, BOOT_CAUSE_POR)

    def test_info_reads_correct_defaults(self):
        """info command reads G2_ID, VERSION correctly."""
        gid = self.b.read(G2_ID.offset)
        assert gid == 0x4732_0001
        ver = self.b.read(G2_VERSION.offset)
        assert ver == 0x0001_0000

    def test_boot_cause_por(self):
        """BOOT_CAUSE shows POR after power-on."""
        bc = read_boot_cause(self.b)
        assert "POR" in bc["causes"]

    def test_queue_submit_nop(self):
        """Stage NOP descriptor and ring doorbell."""
        desc = pack_nop()
        stage_and_doorbell(self.b, desc, queue=0)
        # In SimBackend, doorbell write is stored (WO)
        # Status won't change (no RTL), but the write should not error

    def test_tc0_ctrl_enable(self):
        """TC0 CTRL should show ENABLE=1 by default."""
        ctrl = self.b.read(TC0_CTRL.offset)
        assert ctrl & 1 == 1

    def test_tc0_runstate_idle(self):
        """TC0 RUNSTATE should be IDLE (0) at start."""
        rs = self.b.read(TC0_RUNSTATE.offset)
        assert (rs & 0x7) == 0

    def test_irq_all_masked(self):
        """IRQ_MASK should be all-masked after reset."""
        st = read_irq_status(self.b)
        assert st["mask"] == 0xFFFF_FFFF
        assert st["active"] == 0

    def test_irq_set_and_clear(self):
        """Set IRQ pending, then W1C clear."""
        # Simulate HW setting DESC_DONE
        self.b.set_pending(IRQ_PENDING.offset, 1 << IrqBit.DESC_DONE)
        st = read_irq_status(self.b)
        assert "DESC_DONE" in st["sources"]

        # Clear
        clear_all_irq(self.b)
        st2 = read_irq_status(self.b)
        assert st2["pending"] == 0

    def test_fault_set_and_clear(self):
        """Set TC0 fault, then clear."""
        self.b.set_ro(TC0_FAULT_STATUS.offset, 0x02)  # CRC fault
        fault = self.b.read(TC0_FAULT_STATUS.offset)
        assert fault == 0x02

        # TC0_FAULT_STATUS is W1C
        self.b.write(TC0_FAULT_STATUS.offset, 0x02)
        fault2 = self.b.read(TC0_FAULT_STATUS.offset)
        assert fault2 == 0

    def test_oom_normal_at_start(self):
        """OOM state should be NORMAL at start."""
        st = read_oom_status(self.b)
        assert st["state"] == "NORMAL"
        assert st["usage"] == 0

    def test_trace_empty_at_start(self):
        """Trace ring should be empty at start."""
        st = read_trace_status(self.b)
        assert st["head"] == 0
        assert st["tail"] == 0

    def test_full_host_flow(self, capsys):
        """Full flow: info -> submit -> status -> irq -> clear."""
        # 1. Info
        cmd_info(self.b)
        captured = capsys.readouterr()
        assert "4732" in captured.out

        # 2. Submit NOP
        desc = pack_nop()
        stage_and_doorbell(self.b, desc, queue=0)

        # 3. Check TC status
        rs = self.b.read(TC0_RUNSTATE.offset)
        assert isinstance(rs, int)

        # 4. Simulate IRQ
        self.b.set_pending(IRQ_PENDING.offset, 1 << IrqBit.DESC_DONE)
        st = read_irq_status(self.b)
        assert "DESC_DONE" in st["sources"]

        # 5. Clear
        clear_all_irq(self.b)
        st2 = read_irq_status(self.b)
        assert st2["pending"] == 0
