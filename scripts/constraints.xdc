# constraints.xdc — Timing constraints for ORBIT-G1 on Arty A7-100T
# Target device: xc7a100tcsg324-1
# Clock target : 150 MHz (6.667 ns)
#   Arty A7-100T has a 100 MHz oscillator; 150 MHz via PLL.

create_clock -period 6.667 -name clk [get_ports clk]

set_input_delay  -clock clk -max 1.0 [all_inputs]
set_input_delay  -clock clk -min 0.2 [all_inputs]
set_output_delay -clock clk -max 1.0 [all_outputs]
set_output_delay -clock clk -min 0.2 [all_outputs]

set_false_path -from [get_ports rst_n]
