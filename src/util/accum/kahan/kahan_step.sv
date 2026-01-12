`ifndef __KAHAN_STEP_SV__
`define __KAHAN_STEP_SV__

`include "../parametrizable-floating-point-verilog/floating_point_adder.v"

module kahan_step #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I // // 1 for sign bit
)(
  input                          clk_i,
  input                          rst_ni,
  input  logic [BIT_WIDTH_I-1:0] elem_i,
  input  logic [BIT_WIDTH_I-1:0] c_i,
  input  logic [BIT_WIDTH_I-1:0] sum_i,
  output logic [BIT_WIDTH_I-1:0] c_o,
  output logic [BIT_WIDTH_I-1:0] sum_o
);

  // ===============================
  // Stage 1: Compute y = elem - c
  // ===============================
  logic [BIT_WIDTH_I-1:0] y_comb;
  logic y_underflow_flag, y_overflow_flag, y_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_y (
  //   .a(elem_i),
  //   .b(c_i),
  //   .subtract(1'b1),
  //   .out(y_comb),
  //   .underflow_flag(y_underflow_flag),
  //   .overflow_flag(y_overflow_flag),
  //   .invalid_operation_flag(y_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign y_comb = elem_i - c_i;

  logic [BIT_WIDTH_I-1:0] y_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      y_r <= '0;
    else
      y_r <= y_comb;
  end

  // ===============================
  // Stage 2: Compute t = sum + y
  // ===============================
  logic [BIT_WIDTH_I-1:0] t_comb;
  logic t_underflow_flag, t_overflow_flag, t_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_t (
  //   .a(sum_i),
  //   .b(y_r),
  //   .subtract(1'b0),
  //   .out(t_comb),
  //   .underflow_flag(t_underflow_flag),
  //   .overflow_flag(t_overflow_flag),
  //   .invalid_operation_flag(t_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign t_comb = sum_i + y_r;

  logic [BIT_WIDTH_I-1:0] t_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      t_r <= '0;
    else
      t_r <= t_comb;
  end

  // ===============================
  // Stage 3: Compute t - sum
  // ===============================
  logic [BIT_WIDTH_I-1:0] t_minus_sum_comb;
  logic tms_underflow_flag, tms_overflow_flag, tms_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_t_minus_sum (
  //   .a(t_r),
  //   .b(sum_i),
  //   .subtract(1'b1),
  //   .out(t_minus_sum_comb),
  //   .underflow_flag(tms_underflow_flag),
  //   .overflow_flag(tms_overflow_flag),
  //   .invalid_operation_flag(tms_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign t_minus_sum_comb = t_r - sum_i;

  logic [BIT_WIDTH_I-1:0] t_minus_sum_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      t_minus_sum_r <= '0;
    else
      t_minus_sum_r <= t_minus_sum_comb;
  end

  // ===============================
  // Stage 4: Compute c = (t-sum) - y
  // ===============================
  logic [BIT_WIDTH_I-1:0] c_comb;
  logic c_underflow_flag, c_overflow_flag, c_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_c (
  //   .a(t_minus_sum_r),
  //   .b(y_r),
  //   .subtract(1'b1),
  //   .out(c_comb),
  //   .underflow_flag(c_underflow_flag),
  //   .overflow_flag(c_overflow_flag),
  //   .invalid_operation_flag(c_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign c_comb = t_minus_sum_r - y_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      c_o   <= '0;
      sum_o <= '0;
    end else begin
      c_o   <= c_comb;
      sum_o <= t_r;
    end
  end

endmodule

`endif