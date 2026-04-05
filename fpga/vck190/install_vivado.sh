#!/bin/bash
# install_vivado.sh — Vivado ML 2024.2 Batch Install for ORBIT-G2
#
# Prerequisites:
#   1. AMD account at https://www.amd.com/en/registration/create-account.html
#   2. Download the Linux Web Installer:
#      https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2024-2.html
#      → "AMD Unified Installer for FPGAs & Adaptive SoCs 2024.2: Linux Self Extracting Web Installer"
#      → File: FPGAs_AdaptiveSoCs_Unified_2024.2_1113_1001_Lin64.bin (approx. 300 MB)
#   3. Place the .bin file in this directory (fpga/vck190/)
#
# Usage:
#   chmod +x install_vivado.sh
#   ./install_vivado.sh
#
# Install location: /tools/Xilinx/Vivado/2024.2
# Disk needed: ~70 GB
# Time: 1-3 hours (depending on internet speed)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/tools/Xilinx"
VIVADO_VERSION="2024.2"

# Find installer
INSTALLER=$(find "${SCRIPT_DIR}" -maxdepth 1 -name "FPGAs_AdaptiveSoCs_Unified_*.bin" -o -name "Xilinx_Unified_*.bin" 2>/dev/null | head -1)

if [ -z "${INSTALLER}" ]; then
    echo "=========================================="
    echo "ERROR: Installer .bin not found!"
    echo ""
    echo "Download from:"
    echo "  https://www.xilinx.com/support/download.html"
    echo ""
    echo "Select: Vivado ML Edition - 2024.2"
    echo "  → AMD Unified Installer for FPGAs & Adaptive SoCs 2024.2"
    echo "  → Linux Self Extracting Web Installer (.bin)"
    echo ""
    echo "Place the .bin file in: ${SCRIPT_DIR}/"
    echo "Then re-run this script."
    echo "=========================================="
    exit 1
fi

echo "Found installer: ${INSTALLER}"
echo "Install location: ${INSTALL_DIR}"
echo ""

# Create install config
CONFIG="${SCRIPT_DIR}/vivado_install_config.txt"
cat > "${CONFIG}" << 'EOF'
#### Vivado ML 2024.2 Install Config for ORBIT-G2 ####

# Edition: Vivado ML Enterprise (needed for Versal AI Core / VC1902)
# If no Enterprise license, change to "Vivado ML Standard" and accept
# the device limitation (Versal AI Edge only).
Edition=Vivado ML Enterprise

# Destination
Destination=/tools/Xilinx

# Modules to install
Modules=Vivado:1
Modules=Vitis HLS:0
Modules=Model Composer:0
Modules=Vivado Lab Edition:0

# Devices — install only what we need (saves disk)
# Versal AI Core (VC1902) + Versal Premium (optional)
InstallDevices=Versal AI Core Series:1
InstallDevices=Versal AI Edge Series:0
InstallDevices=Versal Premium Series:0
InstallDevices=Versal HBM Series:0
InstallDevices=Versal Prime Series:0
InstallDevices=Zynq UltraScale+ MPSoC:0
InstallDevices=UltraScale+:0
InstallDevices=UltraScale:0
InstallDevices=7 Series:0
InstallDevices=Spartan-7:0
InstallDevices=Artix-7:0
InstallDevices=Kintex-7:0
InstallDevices=Virtex-7:0

# Accept all licenses
CreateProgramGroupShortcuts=0
EOF

echo "Install config: ${CONFIG}"
echo ""

# Make installer executable
chmod +x "${INSTALLER}"

# Create install directory
sudo mkdir -p "${INSTALL_DIR}"
sudo chown $(whoami) "${INSTALL_DIR}"

echo "Starting Vivado installation..."
echo "This will take 1-3 hours depending on internet speed."
echo ""

# Run installer in batch mode with config
"${INSTALLER}" --agree XilinxEULA,3rdPartyEULA \
    --batch Install \
    --config "${CONFIG}" \
    2>&1 | tee "${SCRIPT_DIR}/vivado_install.log"

echo ""
echo "=========================================="

if [ -f "${INSTALL_DIR}/Vivado/${VIVADO_VERSION}/bin/vivado" ]; then
    echo "SUCCESS: Vivado installed at ${INSTALL_DIR}/Vivado/${VIVADO_VERSION}"
    echo ""
    echo "Add to PATH:"
    echo "  source ${INSTALL_DIR}/Vivado/${VIVADO_VERSION}/settings64.sh"
    echo ""
    echo "Verify:"
    echo "  vivado -version"
    echo ""
    echo "Next: cd yua-t16/fpga/vck190 && vivado -mode batch -source create_project.tcl"
else
    echo "FAILED: Vivado binary not found after installation."
    echo "Check log: ${SCRIPT_DIR}/vivado_install.log"
fi
echo "=========================================="
