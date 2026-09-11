# kv260_dr1.xdc — ORBIT-DR1 KV260 제약 (PLAN W11)
#
# **out-of-context 합성용이다.** PS 블록 디자인을 붙이면 클럭은 PS 에서 오므로
# create_clock 이 아니라 PS 의 clk_out 에 자동으로 잡힌다 — 그때 이 파일에서
# create_clock 줄을 빼고 BD 제약을 쓴다.
#
# ⚠ 이 파일로 합성/구현을 **돌린 적이 없다**. 타이밍 수치는 미측정이다
#   (docs/FPGA.md).

# ── 클럭 ────────────────────────────────────────────────────────────────
# PLAN W11 목표: 100MHz. 안 나오면 낮추고 **낮췄다고 기록한다** (숨기지 않는다).
create_clock -period 10.000 -name clk [get_ports clk]

# ── 리셋 ────────────────────────────────────────────────────────────────
# resetn 은 AXI 리셋이다. 동기 해제를 쓰지만 인가는 비동기이므로 입력 지연만 넉넉히.
set_input_delay -clock clk -max 2.000 [get_ports resetn]
set_input_delay -clock clk -min 0.000 [get_ports resetn]

# ── AXI4-Lite 슬레이브 인터페이스 ───────────────────────────────────────
# OOC 이므로 실제 핀이 없다. 합성기가 가상 IO 로 잡도록 지연만 준다.
set axi_in  [get_ports -quiet {s_axil_awaddr[*] s_axil_awvalid s_axil_wdata[*] \
                               s_axil_wstrb[*] s_axil_wvalid s_axil_bready \
                               s_axil_araddr[*] s_axil_arvalid s_axil_rready}]
set axi_out [get_ports -quiet {s_axil_awready s_axil_wready s_axil_bresp[*] \
                               s_axil_bvalid s_axil_arready s_axil_rdata[*] \
                               s_axil_rresp[*] s_axil_rvalid}]

if {[llength $axi_in] > 0} {
  set_input_delay -clock clk -max 3.000 $axi_in
  set_input_delay -clock clk -min 0.500 $axi_in
}
if {[llength $axi_out] > 0} {
  set_output_delay -clock clk -max 3.000 $axi_out
  set_output_delay -clock clk -min 0.500 $axi_out
}

# ── 보드 신호 ───────────────────────────────────────────────────────────
set_output_delay -clock clk -max 3.000 [get_ports -quiet {irq_out reset_active}]
set_output_delay -clock clk -min 0.500 [get_ports -quiet {irq_out reset_active}]

# ── BRAM 추론 확인용 메모 ───────────────────────────────────────────────
# state_sram / dr1_scratch / act_sram / wgt_sram 은 BRAM 으로 추론돼야 한다.
# yosys 쪽에서는 $mem_v2 로 확인했다 (scripts/synth_gate.sh --check-mem).
# Vivado 에서는 report_utilization 의 Block RAM Tile 수로 확인한다.
# LUTRAM 으로 풀리면 면적이 폭발하므로 **반드시 본다**.
