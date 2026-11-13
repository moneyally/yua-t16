# create_cpm_ip.tcl — CPM PCIe Endpoint (Gen4 x8) via Block Design
#
# versal_cips 는 IPI(Block Design) 안에서만 설정된다. create_ip 로는 안 된다.
#
# ─────────────────────────────────────────────────────────────────────
# 값의 출처 (SSOT)
#
#   BAR 크기        rtl/pcie_ep_versal.sv  의 BAR0_SIZE / BAR2_SIZE / BAR4_SIZE
#                   tools/orbit_mmap_backend.py 의 EXPECTED_BAR_SIZES
#   벤더/디바이스   tools/orbit_mmap_backend.py 의
#                   ORBIT_G2_VENDOR_ID / ORBIT_G2_DEVICE_ID
#   링크 속도/폭    VCK190 CPM, Gen4 x8
#
# 아래 숫자를 바꾸려면 저 파일들을 같이 고쳐야 한다. 어긋나면
# tests/test_vck190_build_contract.py 가 잡는다.
#
# ─────────────────────────────────────────────────────────────────────
# 주의 — 속성 이름은 Vivado 버전에 따라 다르다
#
#   아래 "값" 은 설계에서 나온 것이라 확실하다. 반면 CONFIG.CPM_CONFIG
#   안의 "속성 이름" 은 versal_cips IP 버전마다 달라진다. 이 스크립트는
#   Vivado 없이 작성·검증되지 않았으므로, 처음 돌릴 때는 반드시
#   아래 검증 절차를 거쳐야 한다.
#
#   1) 이 스크립트를 돌린다.
#   2) 속성 오류가 나면 GUI(IP Integrator)에서 CPM 을 같은 "값" 으로
#      설정한 뒤 `write_bd_tcl -force exported_cpm_bd.tcl` 로 뽑는다.
#   3) 뽑힌 tcl 의 속성 이름을 이 파일에 반영한다.
#
#   값과 이름을 분리해 둔 이유가 이것이다. 값은 고치지 말고 이름만 맞춰라.
# ─────────────────────────────────────────────────────────────────────

puts "INFO: Creating Block Design with CPM PCIe endpoint..."

# ── 설계 상수 ────────────────────────────────────────────────────────
# BAR 바이트 크기 (SSOT)
set BAR0_BYTES 1048576   ;# 1 MiB  — 레지스터
set BAR2_BYTES 2097152   ;# 2 MiB  — SRAM 윈도우
set BAR4_BYTES   65536   ;# 64 KiB — DMA 엔진

# PCIe BAR 구경은 log2 로 인코딩된다. 리터럴로 적고 아래에서 검산한다.
set BAR0_SIZE {20}       ;# log2(1 MiB)
set BAR2_SIZE {21}       ;# log2(2 MiB)
set BAR4_SIZE {16}       ;# log2(64 KiB)

set VENDOR_ID {10EE}     ;# Xilinx
set DEVICE_ID {9038}     ;# ORBIT-G2 (현재 placeholder — 호스트 스택과 같은 값)
set MSIX_VECTORS 16
set MSIX_TABLE_SIZE {00F}  ;# 16 벡터 = N-1 = 0x00F

# 리터럴이 바이트 크기와 맞는지 검산한다. 한쪽만 고치면 여기서 멈춘다.
proc log2int {n} {
    set b 0
    while {$n > 1} { set n [expr {$n >> 1}]; incr b }
    return $b
}
foreach {name bytes literal} [list \
        BAR0 $BAR0_BYTES $BAR0_SIZE \
        BAR2 $BAR2_BYTES $BAR2_SIZE \
        BAR4 $BAR4_BYTES $BAR4_SIZE] {
    set want [log2int $bytes]
    if {$want != $literal} {
        error "$name: log2($bytes) = $want 인데 스크립트는 $literal 로 적혀 있다"
    }
    if {($bytes & ($bytes - 1)) != 0} {
        error "$name: BAR 크기는 2의 거듭제곱이어야 한다 ($bytes)"
    }
}
puts "INFO: BAR apertures verified — BAR0=$BAR0_SIZE BAR2=$BAR2_SIZE BAR4=$BAR4_SIZE"

# ── Block Design ─────────────────────────────────────────────────────
create_bd_design "cpm_bd"
create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips cips_0

# CPM PCIe0 = Endpoint, Gen4 (16.0 GT/s), X8
set cpm_cfg [list \
  CPM_PCIE0_MODES                     DMA \
  CPM_PCIE0_DEVICE_PORT_TYPE          {PCI_Express_Endpoint} \
  CPM_PCIE0_MAX_LINK_SPEED            {16.0_GT/s} \
  CPM_PCIE0_LINK_WIDTH                {X8} \
\
  CPM_PCIE0_PF0_VENDOR_ID             $VENDOR_ID \
  CPM_PCIE0_PF0_DEVICE_ID             $DEVICE_ID \
  CPM_PCIE0_PF0_SUBSYSTEM_VENDOR_ID   $VENDOR_ID \
  CPM_PCIE0_PF0_SUBSYSTEM_ID          $DEVICE_ID \
\
  CPM_PCIE0_PF0_BAR0_ENABLED          {1} \
  CPM_PCIE0_PF0_BAR0_TYPE             {Memory} \
  CPM_PCIE0_PF0_BAR0_64BIT            {1} \
  CPM_PCIE0_PF0_BAR0_PREFETCHABLE     {0} \
  CPM_PCIE0_PF0_BAR0_SIZE             $BAR0_SIZE \
\
  CPM_PCIE0_PF0_BAR2_ENABLED          {1} \
  CPM_PCIE0_PF0_BAR2_TYPE             {Memory} \
  CPM_PCIE0_PF0_BAR2_64BIT            {1} \
  CPM_PCIE0_PF0_BAR2_PREFETCHABLE     {0} \
  CPM_PCIE0_PF0_BAR2_SIZE             $BAR2_SIZE \
\
  CPM_PCIE0_PF0_BAR4_ENABLED          {1} \
  CPM_PCIE0_PF0_BAR4_TYPE             {Memory} \
  CPM_PCIE0_PF0_BAR4_64BIT            {1} \
  CPM_PCIE0_PF0_BAR4_PREFETCHABLE     {0} \
  CPM_PCIE0_PF0_BAR4_SIZE             $BAR4_SIZE \
\
  CPM_PCIE0_PF0_MSIX_ENABLED          {1} \
  CPM_PCIE0_PF0_MSIX_CAP_TABLE_SIZE   $MSIX_TABLE_SIZE \
]

puts "INFO: MSI-X vectors = $MSIX_VECTORS (table size $MSIX_TABLE_SIZE)"

if {[catch {
    set_property -dict [list CONFIG.CPM_CONFIG $cpm_cfg] [get_bd_cells cips_0]
} err]} {
    puts "ERROR: CPM 속성 설정 실패 — 이름이 이 Vivado 버전과 다르다."
    puts "ERROR: $err"
    puts "INFO: 파일 상단의 검증 절차를 따라 GUI 설정 후"
    puts "INFO:   write_bd_tcl -force exported_cpm_bd.tcl"
    puts "INFO: 로 뽑아 속성 '이름' 만 교체하라. '값' 은 그대로 둔다."
    return -code error $err
}

validate_bd_design
save_bd_design

make_wrapper -files [get_files cpm_bd.bd] -top
add_files -norecurse [glob -nocomplain \
    ../../build/vivado/*/orbit_g2_protob.gen/sources_1/bd/cpm_bd/hdl/cpm_bd_wrapper.v]

puts "INFO: CPM Block Design created (Gen4 x8 Endpoint)."
