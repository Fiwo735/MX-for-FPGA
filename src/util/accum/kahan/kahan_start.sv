`ifndef __KAHAN_START_SV__
`define __KAHAN_START_SV__

`include "../kahan/kahan_step.sv"

module kahan_start #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I = 1 + EXP_WIDTH_I + MANT_WIDTH_I
)(
  input                          clk_i,
  input                          rst_ni,

  input  logic [BIT_WIDTH_I-1:0] e0,
  input  logic [BIT_WIDTH_I-1:0] e1,
  input  logic [BIT_WIDTH_I-1:0] e2,
  input  logic [BIT_WIDTH_I-1:0] e3,

  output logic [BIT_WIDTH_I-1:0] sum_a_o,
  output logic [BIT_WIDTH_I-1:0] sum_b_o,
  output logic [BIT_WIDTH_I-1:0] c_a_o,
  output logic [BIT_WIDTH_I-1:0] c_b_o
);

  // ===============================
  // Stage 0: Register inputs
  // ===============================
  logic [BIT_WIDTH_I-1:0] e0_r, e1_r, e2_r, e3_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      e0_r <= '0;
      e1_r <= '0;
      e2_r <= '0;
      e3_r <= '0;
    end else begin
      e0_r <= e0;
      e1_r <= e1;
      e2_r <= e2;
      e3_r <= e3;
    end
  end

  // ===============================
  // Stage 1..N: kahan_step_a
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_a_int, c_a_int;

  kahan_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) kahan_step_a (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(e0_r),
    .c_i({BIT_WIDTH_I{1'b0}}), // initial carry
    .sum_i(e1_r),
    .c_o(c_a_int),
    .sum_o(sum_a_int)
  );

  // ===============================
  // Stage 1..N: kahan_step_b
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_b_int, c_b_int;

  kahan_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) kahan_step_b (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(e2_r),
    .c_i({BIT_WIDTH_I{1'b0}}),
    .sum_i(e3_r),
    .c_o(c_b_int),
    .sum_o(sum_b_int)
  );

  // ===============================
  // Final Stage: Register outputs
  // ===============================
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_a_o <= '0;
      sum_b_o <= '0;
      c_a_o   <= '0;
      c_b_o   <= '0;
    end else begin
      sum_a_o <= sum_a_int;
      sum_b_o <= sum_b_int;
      c_a_o   <= c_a_int;
      c_b_o   <= c_b_int;
    end
  end

endmodule

`endif
