`ifndef __FASTTWOSUM_STEP_SV__
`define __FASTTWOSUM_STEP_SV__

`include "../parametrizable-floating-point-verilog/floating_point_adder.v"

module fasttwosum_step #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I // 1 for sign bit
)(
  input                          clk_i,
  input                          rst_ni,
  input  logic [BIT_WIDTH_I-1:0] elem_i,
  input  logic [BIT_WIDTH_I-1:0] error_i,
  input  logic [BIT_WIDTH_I-1:0] sum_i,
  output logic [BIT_WIDTH_I-1:0] error_o,
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
    if (!rst_ni)
      temp_r <= '0;
    else
      temp_r <= temp_comb;
  end

  // ===============================
  // Stage 2: z = temp - sum
  // ===============================
  logic [BIT_WIDTH_I-1:0] z_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_z (
  //   .a(temp_r),
  //   .b(sum_i),
  //   .subtract(1'b1),
  //   .out(z_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign z_comb = temp_r - sum_i;

  logic [BIT_WIDTH_I-1:0] z_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      z_r <= '0;
    else
      z_r <= z_comb;
  end

  // ===============================
  // Stage 3: temp_minus_z = temp - z
  // ===============================
  logic [BIT_WIDTH_I-1:0] temp_minus_z_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_temp_minus_z (
  //   .a(temp_r),
  //   .b(z_r),
  //   .subtract(1'b1),
  //   .out(temp_minus_z_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign temp_minus_z_comb = temp_r - z_r;

  logic [BIT_WIDTH_I-1:0] temp_minus_z_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      temp_minus_z_r <= '0;
    else
      temp_minus_z_r <= temp_minus_z_comb;
  end

  // ===============================
  // Stage 4: sum_minus_temp_minus_z = sum - temp_minus_z
  // ===============================
  logic [BIT_WIDTH_I-1:0] sum_minus_temp_minus_z_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_sum_minus_temp_minus_z (
  //   .a(sum_i),
  //   .b(temp_minus_z_r),
  //   .subtract(1'b1),
  //   .out(sum_minus_temp_minus_z_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign sum_minus_temp_minus_z_comb = sum_i - temp_minus_z_r;

  logic [BIT_WIDTH_I-1:0] sum_minus_temp_minus_z_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      sum_minus_temp_minus_z_r <= '0;
    else
      sum_minus_temp_minus_z_r <= sum_minus_temp_minus_z_comb;
  end

  // ===============================
  // Stage 5: value_minus_z = elem - z
  // ===============================
  logic [BIT_WIDTH_I-1:0] value_minus_z_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_value_minus_z (
  //   .a(elem_i),
  //   .b(z_r),
  //   .subtract(1'b1),
  //   .out(value_minus_z_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign value_minus_z_comb = elem_i - z_r;

  logic [BIT_WIDTH_I-1:0] value_minus_z_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      value_minus_z_r <= '0;
    else
      value_minus_z_r <= value_minus_z_comb;
  end

  // ===============================
  // Stage 6: temp_sum = sum_minus_temp_minus_z + value_minus_z
  // ===============================
  logic [BIT_WIDTH_I-1:0] temp_sum_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_temp_sum (
  //   .a(sum_minus_temp_minus_z_r),
  //   .b(value_minus_z_r),
  //   .subtract(1'b0),
  //   .out(temp_sum_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign temp_sum_comb = sum_minus_temp_minus_z_r + value_minus_z_r;

  logic [BIT_WIDTH_I-1:0] temp_sum_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      temp_sum_r <= '0;
    else
      temp_sum_r <= temp_sum_comb;
  end

  // ===============================
  // Stage 7: error_o = error_i + temp_sum
  // ===============================
  logic [BIT_WIDTH_I-1:0] error_o_comb;
  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_temp_error (
  //   .a(error_i),
  //   .b(temp_sum_r),
  //   .subtract(1'b0),
  //   .out(error_o_comb),
  //   .underflow_flag(),
  //   .overflow_flag(),
  //   .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign error_o_comb = error_i + temp_sum_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      error_o <= '0;
      sum_o   <= '0;
    end else begin
      error_o <= error_o_comb;
      sum_o   <= temp_r; // final sum = temp
    end
  end

endmodule

`endif
