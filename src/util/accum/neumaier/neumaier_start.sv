`ifndef __NEUMAIER_START_SV__
`define __NEUMAIER_START_SV__

`include "../neumaier/neumaier_step.sv"

module neumaier_start #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I // 1 for sign bit
)(
  input                          clk_i,
  input                          rst_ni,

  input  logic [BIT_WIDTH_I-1:0] e0,
  input  logic [BIT_WIDTH_I-1:0] e1,
  input  logic [BIT_WIDTH_I-1:0] e2,
  input  logic [BIT_WIDTH_I-1:0] e3,

  output logic [BIT_WIDTH_I-1:0] sum_a_o,
  output logic [BIT_WIDTH_I-1:0] sum_b_o,
  output logic [BIT_WIDTH_I-1:0] comp_a_o,
  output logic [BIT_WIDTH_I-1:0] comp_b_o
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
  // Stage 1..N: neumaier_step_a
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_a_int, comp_a_int;

  neumaier_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) neumaier_step_a (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(e0_r),
    .comp_i({BIT_WIDTH_I{1'b0}}), // initial compensation = 0
    .sum_i(e1_r),
    .comp_o(comp_a_int),
    .sum_o(sum_a_int)
  );

  // ===============================
  // Stage 1..N: neumaier_step_b
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_b_int, comp_b_int;

  neumaier_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) neumaier_step_b (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(e2_r),
    .comp_i({BIT_WIDTH_I{1'b0}}), // initial compensation = 0
    .sum_i(e3_r),
    .comp_o(comp_b_int),
    .sum_o(sum_b_int)
  );

  // ===============================
  // Final Stage: Register outputs
  // ===============================
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_a_o  <= '0;
      sum_b_o  <= '0;
      comp_a_o <= '0;
      comp_b_o <= '0;
    end else begin
      sum_a_o  <= sum_a_int;
      sum_b_o  <= sum_b_int;
      comp_a_o <= comp_a_int;
      comp_b_o <= comp_b_int;
    end
  end

endmodule

`endif
