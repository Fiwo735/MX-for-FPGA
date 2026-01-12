`ifndef __NEUMAIER_MERGE_SV__
`define __NEUMAIER_MERGE_SV__

`include "../neumaier/neumaier_step.sv"

module neumaier_merge #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I // 1 for sign bit
)(
  input                          clk_i,
  input                          rst_ni,

  input  logic [BIT_WIDTH_I-1:0] sum_a_i,
  input  logic [BIT_WIDTH_I-1:0] comp_a_i,
  input  logic [BIT_WIDTH_I-1:0] sum_b_i,
  input  logic [BIT_WIDTH_I-1:0] comp_b_i,

  output logic [BIT_WIDTH_I-1:0] sum_o,
  output logic [BIT_WIDTH_I-1:0] comp_o
);

  // ===============================
  // Stage 0: Register inputs
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_a_r, comp_a_r, sum_b_r, comp_b_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_a_r  <= '0;
      comp_a_r <= '0;
      sum_b_r  <= '0;
      comp_b_r <= '0;
    end else begin
      sum_a_r  <= sum_a_i;
      comp_a_r <= comp_a_i;
      sum_b_r  <= sum_b_i;
      comp_b_r <= comp_b_i;
    end
  end

  // ===============================
  // Stage 1: neumaier_step_1
  // ===============================
  logic [BIT_WIDTH_I-1:0] temp_sum_s1, temp_comp_s1;

  neumaier_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) neumaier_step_1 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(sum_a_r),
    .comp_i(comp_a_r),
    .sum_i(sum_b_r),
    .comp_o(temp_comp_s1),
    .sum_o(temp_sum_s1)
  );

  // ===============================
  // Stage 2: Register outputs of step 1
  // ===============================
  logic [BIT_WIDTH_I-1:0] temp_sum_r, temp_comp_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      temp_sum_r  <= '0;
      temp_comp_r <= '0;
    end else begin
      temp_sum_r  <= temp_sum_s1;
      temp_comp_r <= temp_comp_s1;
    end
  end

  // ===============================
  // Stage 3: neumaier_step_2
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_s2, comp_s2;

  neumaier_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) neumaier_step_2 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(temp_sum_r),
    .comp_i(comp_b_r),
    .sum_i(temp_comp_r),
    .comp_o(comp_s2),
    .sum_o(sum_s2)
  );

  // ===============================
  // Stage 4: Register final outputs
  // ===============================
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_o  <= '0;
      comp_o <= '0;
    end else begin
      sum_o  <= sum_s2;
      comp_o <= comp_s2;
    end
  end

endmodule

`endif
