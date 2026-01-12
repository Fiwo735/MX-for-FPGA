`ifndef __NEUMAIER_STEP_SV__
`define __NEUMAIER_STEP_SV__

`include "../parametrizable-floating-point-verilog/floating_point_adder.v"
`include "../common/abs_bigger_equal.sv"

module neumaier_step #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I
)(
  input                          clk_i,
  input                          rst_ni,
  input  logic [BIT_WIDTH_I-1:0] elem_i,
  input  logic [BIT_WIDTH_I-1:0] comp_i,
  input  logic [BIT_WIDTH_I-1:0] sum_i,
  output logic [BIT_WIDTH_I-1:0] comp_o,
  output logic [BIT_WIDTH_I-1:0] sum_o
);

  // ===============================
  // Stage 1: temp = sum + elem
  // ===============================
  logic [BIT_WIDTH_I-1:0] temp_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_temp (
  //   .a(sum_i),
  //   .b(elem_i),
  //   .subtract(1'b0),
  //   .out(temp_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign temp_comb = sum_i + elem_i;

  logic [BIT_WIDTH_I-1:0] temp_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) temp_r <= '0;
    else         temp_r <= temp_comb;
  end

  // ===============================
  // Stage 2: sum_minus_temp = sum - temp_r
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_minus_temp_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_sum_minus_temp (
  //   .a(sum_i),
  //   .b(temp_r),
  //   .subtract(1'b1),
  //   .out(sum_minus_temp_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign sum_minus_temp_comb = sum_i - temp_r;

  logic [BIT_WIDTH_I-1:0] sum_minus_temp_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) sum_minus_temp_r <= '0;
    else         sum_minus_temp_r <= sum_minus_temp_comb;
  end

  // ===============================
  // Stage 3: comp_sum_1 and comp_candidate_1
  // ===============================
  logic [BIT_WIDTH_I-1:0] comp_sum_1_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_comp_sum_1 (
  //   .a(sum_minus_temp_r),
  //   .b(elem_i),
  //   .subtract(1'b0),
  //   .out(comp_sum_1_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign comp_sum_1_comb = sum_minus_temp_r + elem_i;

  logic [BIT_WIDTH_I-1:0] comp_sum_1_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) comp_sum_1_r <= '0;
    else         comp_sum_1_r <= comp_sum_1_comb;
  end

  logic [BIT_WIDTH_I-1:0] comp_candidate_1_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_comp_candidate_1 (
  //   .a(comp_i),
  //   .b(comp_sum_1_r),
  //   .subtract(1'b0),
  //   .out(comp_candidate_1_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign comp_candidate_1_comb = comp_i + comp_sum_1_r;

  logic [BIT_WIDTH_I-1:0] comp_candidate_1_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) comp_candidate_1_r <= '0;
    else         comp_candidate_1_r <= comp_candidate_1_comb;
  end

  // ===============================
  // Stage 4: elem_minus_temp and comp_candidate_2
  // ===============================
  logic [BIT_WIDTH_I-1:0] elem_minus_temp_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_elem_minus_temp (
  //   .a(elem_i),
  //   .b(temp_r),
  //   .subtract(1'b1),
  //   .out(elem_minus_temp_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign elem_minus_temp_comb = elem_i - temp_r;

  logic [BIT_WIDTH_I-1:0] elem_minus_temp_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) elem_minus_temp_r <= '0;
    else         elem_minus_temp_r <= elem_minus_temp_comb;
  end

  logic [BIT_WIDTH_I-1:0] comp_sum_2_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_comp_sum_2 (
  //   .a(elem_minus_temp_r),
  //   .b(sum_i),
  //   .subtract(1'b0),
  //   .out(comp_sum_2_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign comp_sum_2_comb = elem_minus_temp_r + sum_i;

  logic [BIT_WIDTH_I-1:0] comp_sum_2_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) comp_sum_2_r <= '0;
    else         comp_sum_2_r <= comp_sum_2_comb;
  end

  logic [BIT_WIDTH_I-1:0] comp_candidate_2_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_comp_candidate_2 (
  //   .a(comp_i),
  //   .b(comp_sum_2_r),
  //   .subtract(1'b0),
  //   .out(comp_candidate_2_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign comp_candidate_2_comb = comp_i + comp_sum_2_r;

  logic [BIT_WIDTH_I-1:0] comp_candidate_2_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) comp_candidate_2_r <= '0;
    else         comp_candidate_2_r <= comp_candidate_2_comb;
  end

  // ===============================
  // Stage 5: Decision and final outputs
  // ===============================
  logic decision;
  abs_bigger_equal #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) abs_bigger_equal_inst (
    .a_i(sum_i),
    .b_i(elem_i),
    .res_o(decision)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      comp_o <= '0;
      sum_o  <= '0;
    end else begin
      comp_o <= (decision) ? comp_candidate_1_r : comp_candidate_2_r;
      sum_o  <= temp_r;
    end
  end

endmodule

`endif
