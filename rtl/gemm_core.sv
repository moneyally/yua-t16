`timescale 1ns/1ps
`default_nettype none

module gemm_core #(
  parameter int MAX_KT = 256,
  parameter int DMA_BEAT_BYTES = 16
)(
  input  logic clk,
  input  logic rst_n,

  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [63:0] act_addr,
  input  logic [63:0] wgt_addr,
  input  logic [63:0] out_addr,
  input  logic [31:0] Kt,

  output logic        done,

  // DMA Read
  output logic         rd_req_valid,
  input  logic         rd_req_ready,
  output logic [63:0]  rd_req_addr,
  output logic [15:0]  rd_req_len_bytes,
  input  logic         rd_done,

  input  logic         rd_data_valid,
  input  logic [127:0] rd_data,
  output logic         rd_data_ready,
  input  logic         rd_data_last,

  // DMA Write
  output logic         wr_req_valid,
  input  logic         wr_req_ready,
  output logic [63:0]  wr_req_addr,
  output logic [15:0]  wr_req_len_bytes,
  input  logic         wr_done,

  output logic         wr_data_valid,
  input  logic         wr_data_ready,
  output logic [127:0] wr_data,
  output logic         wr_data_last,

  output logic [31:0]  perf_cycles,
  output logic [31:0]  perf_bytes,

  output logic         busy
);

  localparam int ADDR_W    = $clog2(MAX_KT);
  localparam int OUT_BEATS = 64;      // 1024 / 16
  localparam int OUT_BYTES = 1024;

  function automatic logic [31:0] clamp_kt(input logic [31:0] in);
    logic [31:0] mx;
    begin
      mx = MAX_KT;
      clamp_kt = (in > mx) ? mx : in;
    end
  endfunction

  // ----------------------------
  // Command latch
  // ----------------------------
  logic [63:0] act_base_r, wgt_base_r, out_base_r;
  logic [31:0] Kt_r;
  logic [15:0] rd_len_r;

  typedef enum logic [3:0] {
    S_IDLE,
    S_RD_ACT_REQ,
    S_RD_ACT_RECV,
    S_RD_WGT_REQ,
    S_RD_WGT_RECV,
    S_ACC_CLR,
    S_COMPUTE,
    S_POST_MAC,   // drain 1cycle (last MAC happens on entry edge)
    S_LATCH,      // ✅ NEW: latch acc_out_flat -> acc_out_lat
    S_WR_REQ,
    S_WR_SEND,
    S_DONE
  } state_t;

  state_t st, st_n;


  assign cmd_ready = (st == S_IDLE);
  wire   cmd_accept = cmd_valid & cmd_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_base_r <= 64'd0;
      wgt_base_r <= 64'd0;
      out_base_r <= 64'd0;
      Kt_r       <= 32'd0;
      rd_len_r   <= 16'd0;
    end else if (cmd_accept) begin
      act_base_r <= act_addr;
      wgt_base_r <= wgt_addr;
      out_base_r <= out_addr;
      Kt_r       <= clamp_kt(Kt);
      rd_len_r   <= clamp_kt(Kt) * DMA_BEAT_BYTES;
    end
  end

  // ----------------------------
  // Perf
  // ----------------------------
  logic [31:0] cyc_r, bytes_r;
  assign perf_cycles = cyc_r;
  assign perf_bytes  = bytes_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cyc_r   <= 0;
      bytes_r <= 0;
    end else begin
      if (busy) cyc_r <= cyc_r + 1;
      bytes_r <= bytes_r
        + ((rd_data_valid && rd_data_ready) ? DMA_BEAT_BYTES : 0)
        + ((wr_data_valid && wr_data_ready) ? DMA_BEAT_BYTES : 0);
    end
  end

  // ----------------------------
  // SRAM interfaces
  // ----------------------------
  logic act_we, wgt_we, act_re, wgt_re;
  logic [ADDR_W-1:0] act_waddr, wgt_waddr;
  logic [ADDR_W-1:0] act_raddr, wgt_raddr;
  logic [127:0] act_rdata, wgt_rdata;

  act_sram #(.MAX_KT(MAX_KT)) u_act (
    .clk(clk), .rst_n(rst_n),
    .we(act_we), .re(act_re),
    .waddr(act_waddr), .wdata(rd_data),
    .raddr(act_raddr), .rdata(act_rdata)
  );

  wgt_sram #(.MAX_KT(MAX_KT)) u_wgt (
    .clk(clk), .rst_n(rst_n),
    .we(wgt_we), .re(wgt_re),
    .waddr(wgt_waddr), .wdata(rd_data),
    .raddr(wgt_raddr), .rdata(wgt_rdata)
  );

  logic [ADDR_W-1:0] load_idx;

  // ✅ hold read address at "last issued" k to avoid reading past Kt (Kt=1 → addr would become 1 and go X)
  logic [ADDR_W-1:0] rd_k_r;
  // ✅ IMPORTANT: SRAM is sync-read. If we update rd_k_r at posedge, the same posedge read would still see OLD rd_k_r.
  // So, when issuing a new k, drive SRAM raddr directly from issue_k_addr in that cycle.
  logic [ADDR_W-1:0] rd_k_mux;
  assign act_raddr = rd_k_mux;
  assign wgt_raddr = rd_k_mux;
  // ----------------------------
  // MAC array
  // ----------------------------
  logic mac_acc_clr, mac_en;
  logic act_issue, wgt_issue;        // issue_fire exposed (debug)
  logic act_issue_d, wgt_issue_d;    // ✅ 1-cycle delayed valid (SRAM->MAC align)
  logic issue_fire;
  logic mac_en_d1;
  logic mac_en_d2;    

  logic signed [7:0]  a_row [0:15];
  logic signed [7:0]  b_col [0:15];

// MAC internal result as packed flat vector (Icarus-safe — no unpacked array output port)
logic [32*256-1:0] acc_out_mac_flat;

// Unpacked views of MAC result and latched outputs for cocotb/FSM access
logic signed [31:0] acc_out_mac [0:255];  // continuous unpack from acc_out_mac_flat
logic signed [31:0] acc_out_flat [0:255]; // latch of mac result (reset-safe, Icarus-safe)
logic signed [31:0] acc_out_lat  [0:255]; // write-out latch

  genvar i;
  generate
    for (i=0;i<16;i++) begin : ROW_UNPACK
      assign a_row[i] = act_rdata[8*i +: 8];
      assign b_col[i] = wgt_rdata[8*i +: 8];
    end
  endgenerate

mac_array u_mac (
  .clk(clk),
  .rst_n(rst_n),
  .en(mac_en),
  .acc_clr(mac_acc_clr),
  .a_row(a_row),
  .b_col(b_col),
  .acc_out_flat(acc_out_mac_flat)
);

// Unpack acc_out_mac_flat -> acc_out_mac[] via continuous assigns (no always_* needed)
genvar ui;
generate
  for (ui = 0; ui < 256; ui++) begin : UNPACK_MAC
    assign acc_out_mac[ui] = acc_out_mac_flat[ui*32 +: 32];
  end
endgenerate

// --------------------------------------------------
// cocotb / Icarus-safe observable accumulators
// Use generate so each element gets its own always_ff — avoids Icarus
// "constant selects in always_* processes" bug with loop-indexed arrays.
// --------------------------------------------------
genvar zz;
generate
  for (zz = 0; zz < 256; zz++) begin : GEN_ACC_FLAT
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
        acc_out_flat[zz] <= 32'sd0;
      else if (st == S_ACC_CLR)
        acc_out_flat[zz] <= 32'sd0;
      else if (st == S_LATCH)
        acc_out_flat[zz] <= acc_out_mac[zz];
    end
  end
endgenerate

genvar kk;
generate
  for (kk = 0; kk < 256; kk++) begin : GEN_ACC_LAT
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
        acc_out_lat[kk] <= 32'sd0;
      else if (st == S_ACC_CLR)
        acc_out_lat[kk] <= 32'sd0;
      else if (st == S_LATCH && (Kt_r != 0))
        acc_out_lat[kk] <= acc_out_mac[kk];
    end
  end
endgenerate

  // ----------------------------
  // Compute issue counter
  // ----------------------------
  logic [31:0] issue_k_addr;
  logic [31:0] issue_k_mac;
  // Icarus-safe truncation: continuous assign outside always_* avoids
  // "constant selects in always_* processes" warning on part-select.
  logic [ADDR_W-1:0] issue_k_addr_trunc;
  assign issue_k_addr_trunc = issue_k_addr[ADDR_W-1:0];

  // ✅ mac_en = valid (1-cycle after issue_fire, because SRAM rdata becomes stable after the issue edge)
  assign mac_en = mac_en_d2;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    issue_k_addr  <= 0;
    issue_k_mac   <= 0;
    act_issue_d   <= 0;
    wgt_issue_d   <= 0;
    mac_en_d1     <= 0;
    mac_en_d2     <= 0;
    rd_k_r        <= '0;
  end else begin
    // 1-cycle delay: SRAM data valid
    act_issue_d <= issue_fire;
    wgt_issue_d <= issue_fire;

    // 2-cycle delay: MAC result valid
    mac_en_d1 <= act_issue_d & wgt_issue_d;
    mac_en_d2 <= mac_en_d1;

    if (st == S_ACC_CLR) begin
      issue_k_addr <= 0;
      issue_k_mac  <= 0;
      rd_k_r       <= '0;
    end else begin
      if (issue_fire) begin
        rd_k_r       <= issue_k_addr[ADDR_W-1:0];
        issue_k_addr <= issue_k_addr + 1;
      end

      if (mac_en_d2)
        issue_k_mac <= issue_k_mac + 1;
    end
  end
end



  // ----------------------------
  // Write datapath (X-safe + small)
  // ----------------------------
  logic [6:0]   out_beat;
  logic [127:0] wr_data_r;
  logic         wr_valid_r;

  function automatic logic [31:0] x32(input logic signed [31:0] v);
    logic [31:0] t;
    begin
      t = v;
// FIX v0.2: X는 항상 0으로 클리핑
      if ((^t === 1'bx)) x32 = 32'd0;
      else              x32 = t;

    end
  endfunction

  function automatic logic [127:0] pack4(input int base);
    begin
      pack4 = {
        x32(acc_out_lat[base+3]),
        x32(acc_out_lat[base+2]),
        x32(acc_out_lat[base+1]),
        x32(acc_out_lat[base+0])
      };
    end
  endfunction

`ifdef COCOTB_SIM
  initial begin
    out_beat   = 7'd0;
    wr_data_r  = 128'd0;
    wr_valid_r = 1'b0;
  end
`endif

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_beat   <= 7'd0;
      wr_data_r  <= 128'd0;
      wr_valid_r <= 1'b0;
    end else begin
      if (st == S_WR_REQ && wr_req_ready) begin
        out_beat   <= 7'd0;
        wr_data_r  <= pack4(0);
        wr_valid_r <= 1'b1;
      end else if (st == S_WR_SEND) begin
        if (wr_valid_r && wr_data_ready) begin
          if (out_beat == OUT_BEATS-1) begin
            wr_valid_r <= 1'b0;
          end else begin
            out_beat   <= out_beat + 1'b1;
            wr_data_r  <= pack4((out_beat + 1) * 4);
            wr_valid_r <= 1'b1;
          end
        end
      end else begin
        wr_valid_r <= 1'b0;
      end
    end
  end

  assign wr_data       = wr_data_r;
  assign wr_data_valid = (st == S_WR_SEND) && wr_valid_r;
  assign wr_data_last  = (st == S_WR_SEND) && wr_valid_r && (out_beat == OUT_BEATS-1);

  // ----------------------------
  // outputs
  // ----------------------------
  assign rd_data_ready = 1'b1;
  assign busy = (st != S_IDLE && st != S_DONE);
  assign done = (st == S_DONE);

  // ----------------------------
  // FSM sequential
  // ----------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st       <= S_IDLE;
      load_idx <= '0;
    end else begin
      st <= st_n;


      if (st == S_RD_ACT_REQ && rd_req_valid && rd_req_ready) load_idx <= '0;
      else if (st == S_RD_WGT_REQ && rd_req_valid && rd_req_ready) load_idx <= '0;
      else if ((st == S_RD_ACT_RECV || st == S_RD_WGT_RECV) && rd_data_valid && rd_data_ready)
        load_idx <= load_idx + 1'b1;
    end
  end

  // ----------------------------
  // FSM combinational
  // ----------------------------
  always_comb begin
    st_n = st;

    rd_req_valid     = 1'b0;
    rd_req_addr      = 64'd0;
    rd_req_len_bytes = rd_len_r;

    wr_req_valid     = 1'b0;
    wr_req_addr      = out_base_r;
    wr_req_len_bytes = OUT_BYTES;

    act_we    = 1'b0;
    wgt_we    = 1'b0;
    act_re    = 1'b0;
    wgt_re    = 1'b0;

    act_waddr = load_idx;
    wgt_waddr = load_idx;

    mac_acc_clr = 1'b0;
    act_issue   = 1'b0;
    wgt_issue   = 1'b0;
   issue_fire  = 1'b0;
    // default raddr = last issued
    rd_k_mux    = rd_k_r;

    case (st)
      S_IDLE: begin
        if (cmd_valid) st_n = S_RD_ACT_REQ;
      end

      S_RD_ACT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = act_base_r;
        if (rd_req_ready) st_n = S_RD_ACT_RECV;
      end

      S_RD_ACT_RECV: begin
        if (rd_data_valid && rd_data_ready) act_we = 1'b1;
        // ✅ "진짜 마지막 beat" 기준으로 수신 완료 확정
        if (rd_data_valid && rd_data_ready && rd_data_last)
          st_n = S_RD_WGT_REQ;
      end

      S_RD_WGT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = wgt_base_r;
        if (rd_req_ready) st_n = S_RD_WGT_RECV;
      end

      S_RD_WGT_RECV: begin
        if (rd_data_valid && rd_data_ready) wgt_we = 1'b1;
        if (rd_data_valid && rd_data_ready && rd_data_last)
          st_n = S_ACC_CLR;
      end

      S_ACC_CLR: begin
        mac_acc_clr = 1'b1;
        if (Kt_r == 0) st_n = S_POST_MAC;
        else           st_n = S_COMPUTE;
      end

      S_COMPUTE: begin
        // ✅ issue_fire: exactly one pulse per k (bounded)
        issue_fire = (issue_k_addr < Kt_r);
        act_issue  = issue_fire;
        wgt_issue  = issue_fire;

        // ✅ drive SRAM address directly from issue_k_addr in the issue cycle (sync read!)
        // Use issue_k_addr_trunc (continuous assign outside always) to avoid Icarus part-select warning
        rd_k_mux = issue_fire ? issue_k_addr_trunc : rd_k_r;

        // ✅ one-cycle SRAM read enable per issue is enough (rdata holds X-safe in SRAM model)
        act_re = issue_fire;
        wgt_re = issue_fire;

        // ✅ exit after MAC count has reached Kt_r (robust: >=)
        if (issue_k_mac >= Kt_r)
          st_n = S_POST_MAC;
      end

      S_POST_MAC: begin
        // ✅ fixed 1-cycle drain: always move forward
        st_n = S_LATCH;
      end

      S_LATCH: begin
        st_n = S_WR_REQ;
      end

      S_WR_REQ: begin
        wr_req_valid = 1'b1;
        if (wr_req_ready) st_n = S_WR_SEND;
      end

      S_WR_SEND: begin
        if (wr_data_ready && wr_valid_r && (out_beat == OUT_BEATS-1))
          st_n = S_DONE;
      end

      S_DONE: begin
        st_n = S_IDLE;
      end

      default: st_n = S_IDLE;
    endcase
  end

endmodule

`default_nettype wire
