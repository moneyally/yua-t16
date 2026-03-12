# constraints.xdc — Timing constraints for ORBIT-G1 on Arty A7-100T
# Target device: xc7a100tcsg324-1

# Primary clock: 150 MHz (6.667 ns period)
# Arty A7-100T onboard oscillator is 100 MHz; this targets a PLL-generated 150 MHz.
create_clock -period 6.667 -name clk [get_ports clk]

# Input setup/hold (1 ns margin from clock edge)
set_input_delay  -clock clk -max 1.0 [all_inputs]
set_input_delay  -clock clk -min 0.2 [all_inputs]

# Output delay
set_output_delay -clock clk -max 1.0 [all_outputs]
set_output_delay -clock clk -min 0.2 [all_outputs]

# Relax timing on reset (async, multicycle)
set_false_path -from [get_ports rst_n]
