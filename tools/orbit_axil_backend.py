"""
orbit_axil_backend.py — AXI4-Lite backend for ORBIT (cocotb)

`CocotbBackend` 는 `g2_ctrl_top` 의 내부 레지스터 버스(`reg_addr`/`reg_wr_en`)를
직접 흔든다. **보드는 그 버스에 손을 못 댄다** — PS 가 AXI4-Lite 로 들어오고
`axil_reg_bridge` 를 지나야 같은 자리에 닿는다.

이 백엔드는 그 한 칸을 채운다:

    OrbitDevice (tools/orbit_device.py)
      → AxiLiteBackend            ← 여기
        → dr1_soc_top (axil_reg_bridge → g2_ctrl_top → dr1_top)

`CocotbBackend` 와 **같은 인터페이스**다. 그래서 `tb/tb_dr1_host_e2e.py` 가 쓰던
호스트 코드를 한 줄도 안 고치고 보드 최상위에 그대로 붙일 수 있다 —
바뀌는 것은 백엔드 하나뿐이고, 보드에서도 그렇게 바뀐다 (`/dev/mem` 백엔드).

AXI4-Lite 규약 중 여기서 지키는 것:
  - VALID 는 **READY 를 볼 때까지** 유지한다. 한 사이클만 내면 씹힌다
  - ready 는 **falling edge 에서 본다**. rising edge 직후에는 이미 내려가 있다
    (docs/BUGS.md BUG-008a/b — 이 레포가 같은 실수를 네 번 했다)
  - AW 와 W 는 순서가 정해져 있지 않다. 여기서는 **동시에 내보낸다**
"""
from __future__ import annotations

from tools.orbit_backend import Backend

RESP_OKAY = 0


class AxiLiteBackend(Backend):
    """`dr1_soc_top` 의 AXI4-Lite 슬레이브 포트를 흔드는 백엔드.

    동기 `read`/`write` 는 **쓰지 않는다** — AXI 는 여러 사이클이 걸리는
    핸드셰이크라 한 사이클 안에 끝나는 척할 수 없다. `OrbitDevice` 의 동기
    API 를 만족시키려고 남겨 두지만, 값은 마지막 async 접근의 캐시다.
    실제 접근은 `async_read`/`async_write` 로 한다.
    """

    def __init__(self, dut):
        self._dut = dut
        self._clk = dut.clk
        self._last = {}          # offset -> 마지막으로 읽은 값 (동기 API 용)

    # ── 동기 API (호환용) ────────────────────────────────────────
    def read(self, offset: int) -> int:
        return self._last.get(offset, 0)

    def write(self, offset: int, value: int):
        raise RuntimeError(
            "AxiLiteBackend 는 동기 write 를 지원하지 않는다 — "
            "await backend.async_write(...) 를 쓸 것"
        )

    # ── AXI4-Lite 채널 ───────────────────────────────────────────
    async def _drive_until_ready(self, valid_sig, ready_sig, setup):
        """VALID 를 올리고 **falling edge 에서** READY 를 볼 때까지 유지한다."""
        from cocotb.triggers import FallingEdge, RisingEdge
        setup()
        valid_sig.value = 1
        while True:
            await FallingEdge(self._clk)
            if int(ready_sig.value):
                break
        await RisingEdge(self._clk)     # 핸드셰이크가 성립하는 엣지 하나
        valid_sig.value = 0

    async def async_write(self, offset: int, value: int):
        """AW+W 를 내보내고 B 응답까지 받는다."""
        import cocotb
        from cocotb.triggers import FallingEdge, RisingEdge

        d = self._dut

        async def aw():
            await self._drive_until_ready(
                d.s_axil_awvalid, d.s_axil_awready,
                lambda: setattr(d.s_axil_awaddr, "value", offset),
            )

        async def w():
            def setup():
                d.s_axil_wdata.value = value & 0xFFFF_FFFF
                d.s_axil_wstrb.value = 0xF
            await self._drive_until_ready(d.s_axil_wvalid, d.s_axil_wready, setup)

        # AW 와 W 를 **따로 굴린다**. 한쪽 ready 가 늦어도 다른 쪽이 막히지 않는다
        t_aw = cocotb.start_soon(aw())
        t_w = cocotb.start_soon(w())
        await t_aw
        await t_w

        d.s_axil_bready.value = 1
        while True:
            await FallingEdge(self._clk)
            if int(d.s_axil_bvalid.value):
                resp = int(d.s_axil_bresp.value)
                break
        await RisingEdge(self._clk)
        d.s_axil_bready.value = 0
        assert resp == RESP_OKAY, f"AXI-Lite 쓰기 BRESP={resp} (offset={offset:#x})"

    async def async_read(self, offset: int) -> int:
        """AR 을 내보내고 R 데이터를 받는다."""
        from cocotb.triggers import FallingEdge, RisingEdge

        d = self._dut
        await self._drive_until_ready(
            d.s_axil_arvalid, d.s_axil_arready,
            lambda: setattr(d.s_axil_araddr, "value", offset),
        )

        d.s_axil_rready.value = 1
        while True:
            await FallingEdge(self._clk)
            if int(d.s_axil_rvalid.value):
                val = int(d.s_axil_rdata.value)
                resp = int(d.s_axil_rresp.value)
                break
        await RisingEdge(self._clk)
        d.s_axil_rready.value = 0
        assert resp == RESP_OKAY, f"AXI-Lite 읽기 RRESP={resp} (offset={offset:#x})"

        self._last[offset] = val
        return val

    # ── 수명주기 ─────────────────────────────────────────────────
    async def reset(self):
        """리셋을 걸고 AXI 채널을 전부 내려 둔다."""
        from cocotb.triggers import RisingEdge, Timer

        d = self._dut
        d.resetn.value = 0
        for sig, val in (
            ("s_axil_awaddr", 0), ("s_axil_awvalid", 0),
            ("s_axil_wdata", 0), ("s_axil_wstrb", 0), ("s_axil_wvalid", 0),
            ("s_axil_bready", 0),
            ("s_axil_araddr", 0), ("s_axil_arvalid", 0), ("s_axil_rready", 0),
        ):
            getattr(d, sig).value = val
        await Timer(100, unit="ns")
        d.resetn.value = 1
        for _ in range(30):
            await RisingEdge(self._clk)
            if int(d.reset_active.value) == 0:
                break
        await RisingEdge(self._clk)

    async def tick(self, n: int = 1):
        from cocotb.triggers import RisingEdge
        for _ in range(n):
            await RisingEdge(self._clk)

    async def enqueue_and_tick(self, dev, desc: bytes, queue: int = 0,
                               wait_cycles: int = 50):
        """디스크립터를 스테이징하고 도어벨을 친다 — `CocotbBackend` 와 같은 절차."""
        from tools.orbit_desc import desc_to_words
        from tools.orbit_mmio_map import DESC_STAGE_BASE, BASE, QUEUE_DOORBELLS

        words = desc_to_words(desc)
        stage_off = DESC_STAGE_BASE - BASE
        for i, w in enumerate(words):
            await self.async_write(stage_off + i * 4, w)
        await self.async_write(QUEUE_DOORBELLS[queue].offset, 0x0001)
        await self.tick(wait_cycles)
