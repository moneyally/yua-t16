module ctrl_fsm #(
  parameter DESC_SIZE = 64
)(
  input  logic        clk,
  input  logic        rst_n,

  // Descriptor input (from host / DMA / FIFO)
  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:DESC_SIZE-1],
  output logic        desc_ready,

  // GEMM dispatch interface (to compute core)
  output logic        gemm_valid,
  output logic [63:0] act_addr,
  output logic [63:0] wgt_addr,
  output logic [63:0] out_addr,
  output logic [31:0] Kt,
  input  logic        gemm_done
);

  // ----------------------------
  // Descriptor header fields
  // ----------------------------
  logic [7:0]  desc_type;
  logic [7:0]  desc_flags;
  logic [31:0] desc_length;
  logic [63:0] desc_next;

  // ----------------------------
  // FSM states
  // ----------------------------
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_DECODE,
    ST_DISPATCH,
    ST_WAIT,
    ST_DONE
  } state_t;

  state_t state, state_n;

  // ----------------------------
  // Descriptor parsing (little-endian)
  // ----------------------------
  always_comb begin
    desc_type   = desc_bytes[0];
    desc_flags  = desc_bytes[1];

    desc_length = {
      desc_bytes[7], desc_bytes[6],
      desc_bytes[5], desc_bytes[4]
    };

    desc_next = {
      desc_bytes[15], desc_bytes[14], desc_bytes[13], desc_bytes[12],
      desc_bytes[11], desc_bytes[10], desc_bytes[9],  desc_bytes[8]
    };

    // GEMM fields
    act_addr = {
      desc_bytes[23], desc_bytes[22], desc_bytes[21], desc_bytes[20],
      desc_bytes[19], desc_bytes[18], desc_bytes[17], desc_bytes[16]
    };

    wgt_addr = {
      desc_bytes[31], desc_bytes[30], desc_bytes[29], desc_bytes[28],
      desc_bytes[27], desc_bytes[26], desc_bytes[25], desc_bytes[24]
    };

    out_addr = {
      desc_bytes[39], desc_bytes[38], desc_bytes[37], desc_bytes[36],
      desc_bytes[35], desc_bytes[34], desc_bytes[33], desc_bytes[32]
    };

    Kt = {
      desc_bytes[43], desc_bytes[42],
      desc_bytes[41], desc_bytes[40]
    };
  end

  // ----------------------------
  // FSM sequential
  // ----------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= ST_IDLE;
    else
      state <= state_n;
  end

  // ----------------------------
  // FSM combinational
  // ----------------------------
  always_comb begin
    // defaults
    state_n     = state;
    desc_ready  = 1'b0;
    gemm_valid  = 1'b0;

    case (state)
      ST_IDLE: begin
        desc_ready = 1'b1;
        if (desc_valid)
          state_n = ST_DECODE;
      end

      ST_DECODE: begin
        if (desc_type == 8'h02)  // GEMM_INT8
          state_n = ST_DISPATCH;
        else
          state_n = ST_DONE; // unsupported descriptor (v1)
      end

      ST_DISPATCH: begin
        gemm_valid = 1'b1;
        state_n = ST_WAIT;
      end

      ST_WAIT: begin
        if (gemm_done)
          state_n = ST_DONE;
      end

      ST_DONE: begin
        state_n = ST_IDLE;
      end

      default: state_n = ST_IDLE;
    endcase
  end

endmodule
