module gemm_stub (
  input  logic        clk,
  input  logic        rst_n,

  input  logic        valid,
  input  logic [31:0] Kt,

  output logic        done
);

  logic [31:0] counter;
  logic        busy;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy    <= 1'b0;
      counter <= 32'd0;
      done    <= 1'b0;
    end else begin
      done <= 1'b0;

      if (valid && !busy) begin
        busy    <= 1'b1;
        counter <= Kt;   // simulate K-loop latency
      end else if (busy) begin
        if (counter > 0)
          counter <= counter - 1;
        else begin
          busy <= 1'b0;
          done <= 1'b1;
        end
      end
    end
  end

endmodule
