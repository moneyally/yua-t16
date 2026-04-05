# create_cpm_ip.tcl — CPM PCIe Endpoint via Block Design
# versal_cips requires IPI (Block Design). Cannot use create_ip standalone.
#
# This script creates a Block Design with CIPS CPM endpoint configured,
# then generates a wrapper for use as top-level.

puts "INFO: Creating Block Design with CPM PCIe endpoint..."

# Create block design
create_bd_design "cpm_bd"

# Add CIPS IP
create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips cips_0

# Configure CPM for PCIe DMA endpoint Gen4 x8
# NOTE: Property names are version-specific.
# If this fails, use Vivado GUI → IP Integrator to configure, then export Tcl.
set_property -dict [list \
  CONFIG.CPM_CONFIG { \
    CPM_PCIE0_MODES DMA \
    CPM_PCIE0_MAX_LINK_SPEED 16.0_GT/s \
    CPM_PCIE0_LINK_WIDTH X8 \
    CPM_PCIE0_PF0_BAR0_QDMA_64BIT 1 \
    CPM_PCIE0_PF0_BAR0_QDMA_PREFETCHABLE 0 \
    CPM_PCIE0_PF0_BAR0_QDMA_SCALE Megabytes \
    CPM_PCIE0_PF0_BAR0_QDMA_SIZE 1 \
  } \
] [get_bd_cells cips_0]

# Validate and save
validate_bd_design
save_bd_design

# Generate wrapper
make_wrapper -files [get_files cpm_bd.bd] -top
add_files -norecurse [glob -nocomplain ../../build/vivado/*/orbit_g2_protob.gen/sources_1/bd/cpm_bd/hdl/cpm_bd_wrapper.v]

puts "INFO: CPM Block Design created."
puts "INFO: If property errors occur, use Vivado GUI to configure CPM manually."
puts "INFO: Then: write_bd_tcl -force exported_cpm_bd.tcl"
