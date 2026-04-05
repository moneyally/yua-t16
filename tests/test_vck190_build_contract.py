"""test_vck190_build_contract.py — VCK190 build artifacts consistency.

Verifies that Tcl scripts, XDC, and docs are present and internally consistent.
"""
import os
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FPGA_DIR = os.path.join(PROJECT_ROOT, "fpga", "vck190")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


class TestBuildFilesExist:
    def test_create_project_tcl(self):
        assert os.path.isfile(os.path.join(FPGA_DIR, "create_project.tcl"))

    def test_create_cpm_ip_tcl(self):
        assert os.path.isfile(os.path.join(FPGA_DIR, "create_cpm_ip.tcl"))

    def test_xdc(self):
        assert os.path.isfile(os.path.join(FPGA_DIR, "vck190_pcie.xdc"))

    def test_first_smoke_script(self):
        assert os.path.isfile(os.path.join(SCRIPTS_DIR, "vck190_first_smoke.py"))

    def test_build_doc(self):
        assert os.path.isfile(os.path.join(DOCS_DIR, "ORBIT_G2_VCK190_BUILD.md"))

    def test_bringup_doc(self):
        assert os.path.isfile(os.path.join(DOCS_DIR, "ORBIT_G2_VCK190_PCIE_BRINGUP.md"))

    def test_failure_matrix_doc(self):
        assert os.path.isfile(os.path.join(DOCS_DIR, "ORBIT_G2_VCK190_FAILURE_MATRIX.md"))


class TestTclContent:
    def _read(self, name):
        with open(os.path.join(FPGA_DIR, name)) as f:
            return f.read()

    def test_project_tcl_targets_vck190(self):
        content = self._read("create_project.tcl")
        assert "xcvc1902" in content
        assert "vck190" in content

    def test_project_tcl_sets_top(self):
        content = self._read("create_project.tcl")
        assert "g2_protob_top" in content

    def test_cpm_tcl_gen4_x8(self):
        content = self._read("create_cpm_ip.tcl")
        assert "16.0_GT/s" in content or "Gen4" in content.lower()
        assert "X8" in content

    def test_cpm_tcl_bar_sizes(self):
        """BAR size encoding: BAR0=20 (1M), BAR2=21 (2M), BAR4=16 (64K)."""
        content = self._read("create_cpm_ip.tcl")
        # BAR0 size = 20 (log2 of 1 MiB)
        assert "BAR0_SIZE" in content and "{20}" in content
        # BAR2 size = 21
        assert "BAR2_SIZE" in content and "{21}" in content
        # BAR4 size = 16
        assert "BAR4_SIZE" in content and "{16}" in content

    def test_cpm_tcl_vendor_device_id(self):
        content = self._read("create_cpm_ip.tcl")
        assert "10EE" in content  # vendor
        assert "9038" in content  # device

    def test_cpm_tcl_msix(self):
        content = self._read("create_cpm_ip.tcl")
        assert "MSIX_ENABLED" in content
        assert "16" in content or "00F" in content  # 16 vectors = 0x00F

    def test_cpm_tcl_endpoint(self):
        content = self._read("create_cpm_ip.tcl")
        assert "Endpoint" in content


class TestXdcContent:
    def test_xdc_references_vck190(self):
        with open(os.path.join(FPGA_DIR, "vck190_pcie.xdc")) as f:
            content = f.read()
        assert "VCK190" in content
        assert "GTY103" in content or "GTY104" in content

    def test_xdc_mentions_refclk_pins(self):
        with open(os.path.join(FPGA_DIR, "vck190_pcie.xdc")) as f:
            content = f.read()
        assert "W39" in content  # GTY103 REFCLK0 P
        assert "R39" in content  # GTY104 REFCLK0 P


class TestSmokeScriptContent:
    def test_smoke_script_reads_g2_id(self):
        with open(os.path.join(SCRIPTS_DIR, "vck190_first_smoke.py")) as f:
            content = f.read()
        assert "G2_ID" in content
        assert "0x4732_0001" in content

    def test_smoke_uses_orbit_device(self):
        with open(os.path.join(SCRIPTS_DIR, "vck190_first_smoke.py")) as f:
            content = f.read()
        assert "OrbitDevice" in content
        assert "LinuxPciBarOpener" in content
        assert "MmapBackend" in content


class TestBarConsistency:
    """BAR size parameters are consistent across all files."""

    def test_bar0_1mib_everywhere(self):
        from tools.orbit_mmap_backend import EXPECTED_BAR_SIZES
        assert EXPECTED_BAR_SIZES[0] == 1048576  # 1 MiB

        with open(os.path.join(FPGA_DIR, "create_cpm_ip.tcl")) as f:
            assert "{20}" in f.read()  # 2^20 = 1 MiB

    def test_bar4_64k_everywhere(self):
        from tools.orbit_mmap_backend import EXPECTED_BAR_SIZES
        assert EXPECTED_BAR_SIZES[4] == 65536  # 64 KiB

        with open(os.path.join(FPGA_DIR, "create_cpm_ip.tcl")) as f:
            assert "{16}" in f.read()  # 2^16 = 64 KiB
