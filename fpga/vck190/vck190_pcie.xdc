## vck190_pcie.xdc — VCK190 PCIe Constraints for ORBIT-G2 Proto-B
## SSOT: ORBIT_G2_VCK190_PCIE_BRINGUP.md, UG1366
##
## Target: VCK190 / VC1902
## PCIe: Gen4 x8 via GTY103/GTY104, card edge connector P3
##
## NOTE: CPM (hardened PCIe) handles GT placement internally.
## These constraints cover refclk, reset, and board-level I/O only.
## CPM internal GT lanes do NOT need LOC constraints — they are fixed in silicon.

## ═══════════════════════════════════════════════════════════════
## PCIe Reference Clock
## ═══════════════════════════════════════════════════════════════
## Source: 100 MHz from PCIe card edge → U39 (1:2 LVDS buffer)
## U39 Q0 → GTY103 REFCLK0: W39(P) / W40(N)
## U39 Q1 → GTY104 REFCLK0: R39(P) / R40(N)
##
## CPM uses these refclks internally. If the design uses PL PCIE
## (which we don't), explicit clock constraints would be needed.
## For CPM: the refclk is routed in silicon. No explicit LOC needed.
##
## Confirm with: report_property [get_ports *pcie*refclk*] after IP generation.

## ═══════════════════════════════════════════════════════════════
## PCIe PERST# (Reset)
## ═══════════════════════════════════════════════════════════════
## PCIe card edge PERST# → VCK190 board → ACAP pin
## Pin assignment: 확인 필요 — depends on board schematic net name
## Typically named sys_rst_n or pcie_perstn
##
## set_property PACKAGE_PIN <PIN> [get_ports pcie_perstn]
## set_property IOSTANDARD LVCMOS18 [get_ports pcie_perstn]
##
## TODO: Confirm PERST# pin from VCK190 schematic.
## VCK190 PCIe TRD uses CPM which handles PERST internally.
## If CPM manages PERST, no explicit constraint needed.

## ═══════════════════════════════════════════════════════════════
## Timing
## ═══════════════════════════════════════════════════════════════
## CPM user_clk is generated internally (250 MHz for Gen4 x8).
## Create a clock constraint if the user logic clock is different.
##
## For Proto-B skeleton: all logic runs on CPM user_clk.
## No additional clock constraints needed unless PL clocking is added.

## create_clock -period 4.000 -name user_clk [get_pins <cpm_user_clk_pin>]
## TODO: Add after IP generation reveals exact pin path.

## ═══════════════════════════════════════════════════════════════
## False paths
## ═══════════════════════════════════════════════════════════════
## Reset synchronizer crossing (reset_seq) — async reset input
## set_false_path -from [get_ports pcie_perstn]

## ═══════════════════════════════════════════════════════════════
## Board I/O (non-PCIe)
## ═══════════════════════════════════════════════════════════════
## No additional board I/O used in Proto-B first-smoke.
## Debug LEDs/UART can be added later.

## ═══════════════════════════════════════════════════════════════
## Summary
## ═══════════════════════════════════════════════════════════════
## Confirmed:
##   - CPM handles GT placement, refclk, link training internally
##   - No explicit LOC for GTY103/GTY104 lanes
##   - Refclk: W39/W40 (GTY103), R39/R40 (GTY104) — board-level
##
## 확인 필요:
##   - PERST# pin (may be CPM-internal on VCK190)
##   - user_clk pin path (from IP generation)
##   - Any board-level pull-up/pull-down requirements
