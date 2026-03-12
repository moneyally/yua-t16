#!/usr/bin/env bash
# scripts/run_openlane.sh — Run OpenLane flow for gemm_int4 or gemm_wb
#
# Usage:
#   ./scripts/run_openlane.sh gemm_int4   # GEMM tile standalone
#   ./scripts/run_openlane.sh gemm_wb     # GEMM + Wishbone wrapper
#
# Prerequisites:
#   docker pull efabless/openlane:2023.07.19-1
#   volare enable --pdk sky130 78b7bc32ddb4b6f14f76883c2e2dc5b5de9d1cbc --pdk-root /home/dmsal020813/project/pdks
#
set -e

DESIGN=${1:-gemm_int4}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PDK_ROOT=${PDK_ROOT:-/home/dmsal020813/project/pdks}
OPENLANE_IMAGE="efabless/openlane:2023.07.19-1"
OPENLANE_ROOT=${OPENLANE_ROOT:-/home/dmsal020813/project/openlane_src}

echo "===== ORBIT-G1 OpenLane Flow ====="
echo "Design    : $DESIGN"
echo "Repo root : $REPO_ROOT"
echo "PDK root  : $PDK_ROOT"
echo "Image     : $OPENLANE_IMAGE"
echo "=================================="

# Check PDK exists
if [ ! -d "$PDK_ROOT/sky130A" ]; then
    echo "ERROR: PDK not found at $PDK_ROOT/sky130A"
    echo "Run: volare enable --pdk sky130 78b7bc32ddb4b6f14f76883c2e2dc5b5de9d1cbc --pdk-root $PDK_ROOT"
    exit 1
fi

# Check design config
DESIGN_DIR="$REPO_ROOT/openlane/$DESIGN"
if [ ! -f "$DESIGN_DIR/config.json" ]; then
    echo "ERROR: $DESIGN_DIR/config.json not found"
    exit 1
fi

echo "Starting OpenLane Docker container..."
docker run --rm \
    -v "$REPO_ROOT:/yua-t16" \
    -v "$PDK_ROOT:/home/user/pdk" \
    -e PDK_ROOT=/home/user/pdk \
    -e PDK=sky130A \
    -e STD_CELL_LIBRARY=sky130_fd_sc_hd \
    -u "$(id -u):$(id -g)" \
    "$OPENLANE_IMAGE" \
    bash -c "
        cd /yua-t16/openlane/$DESIGN && \
        python3 /openLANE_flow/flow.tcl \
            -design /yua-t16/openlane/$DESIGN \
            -tag run_\$(date +%Y%m%d_%H%M) \
            -overwrite \
            2>&1 | tee /yua-t16/openlane/$DESIGN/flow.log
    "

echo ""
echo "===== Flow complete ====="
echo "Results in: openlane/$DESIGN/runs/"
echo "Key files:"
echo "  runs/*/reports/synthesis/*.rpt   — synthesis stats"
echo "  runs/*/reports/routing/*.rpt     — routing stats"
echo "  runs/*/results/final/gds/*.gds   — final layout"
