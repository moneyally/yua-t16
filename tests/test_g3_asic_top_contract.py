"""test_g3_asic_top_contract.py — ASIC top structural contract verification.

Validates that g3_asic_top.sv has required sections, ports, modules, and domains.
"""
import os
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOP_PATH = os.path.join(os.path.dirname(__file__), "..", "rtl", "g3_asic_top.sv")


@pytest.fixture
def top_source():
    with open(TOP_PATH) as f:
        return f.read()


class TestRequiredPorts:
    REQUIRED_PORTS = [
        "core_clk", "mem_clk", "pcie_clk", "fabric_clk", "por_n",
        "reg_addr", "reg_wr_en", "reg_wr_data", "reg_rd_data",
        "irq_out", "reset_active",
        "hbm_req_valid", "hbm_req_ready", "hbm_req_addr",
        "hbm_rsp_valid", "hbm_rsp_rdata",
        "fabric_tx_valid", "fabric_tx_ready", "fabric_tx_data", "fabric_tx_last",
        "fabric_rx_valid", "fabric_rx_ready", "fabric_rx_data", "fabric_rx_last",
        "chip_status",
    ]

    @pytest.mark.parametrize("port", REQUIRED_PORTS)
    def test_port_exists(self, top_source, port):
        assert port in top_source, f"Required port '{port}' not found in g3_asic_top.sv"


class TestRequiredClockDomains:
    def test_core_clk(self, top_source):
        assert "core_clk" in top_source

    def test_mem_clk(self, top_source):
        assert "mem_clk" in top_source

    def test_pcie_clk(self, top_source):
        assert "pcie_clk" in top_source

    def test_fabric_clk(self, top_source):
        assert "fabric_clk" in top_source


class TestRequiredSections:
    SECTIONS = [
        "Control Plane",
        "Compute Cluster",
        "Training",
        "Distributed",
        "Memory",
    ]

    @pytest.mark.parametrize("section", SECTIONS)
    def test_section_exists(self, top_source, section):
        assert section.upper() in top_source.upper(), \
            f"Section '{section}' not found in g3_asic_top.sv"


class TestRequiredModules:
    def test_g3_reg_top_instantiated(self, top_source):
        assert "g3_reg_top" in top_source

    def test_scale_fabric_ctrl_instantiated(self, top_source):
        assert "scale_fabric_ctrl" in top_source

    def test_hbm_stub_present(self, top_source):
        assert "HBM" in top_source.upper() and "stub" in top_source.lower()


class TestRequiredTodos:
    """Future work should be marked as TODO."""
    FUTURE_ITEMS = [
        "noc_router",
        "multi_chip_reset",
        "hbm_ctrl",
        "xla_desc_decoder",
        "CDC",
    ]

    @pytest.mark.parametrize("item", FUTURE_ITEMS)
    def test_todo_marked(self, top_source, item):
        # Item should appear in the file (as TODO or instantiation)
        assert item.lower() in top_source.lower(), \
            f"Future item '{item}' not referenced in g3_asic_top.sv"


class TestG2Compatibility:
    def test_no_g2_address_mutation(self, top_source):
        """g3_asic_top should not redefine G2 register addresses."""
        assert "0x8030" not in top_source, \
            "g3_asic_top should not contain G2 address literals (delegated to reg_top)"

    def test_reg_top_delegation(self, top_source):
        """Register access should go through g3_reg_top, not direct decode."""
        assert "g3_reg_top" in top_source
        # Should NOT have its own case/switch on addresses
        assert "case (reg_addr)" not in top_source


class TestMemoryBoundary:
    def test_hbm_channels(self, top_source):
        assert "NUM_HBM_CH" in top_source

    def test_hbm_req_interface(self, top_source):
        for sig in ["hbm_req_valid", "hbm_req_addr", "hbm_req_wdata"]:
            assert sig in top_source


class TestFabricBoundary:
    def test_fabric_tx_rx(self, top_source):
        for sig in ["fabric_tx_valid", "fabric_rx_valid"]:
            assert sig in top_source
