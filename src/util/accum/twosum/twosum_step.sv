`ifndef __TWOSUM_STEP_SV__
`define __TWOSUM_STEP_SV__

`include "../parametrizable-floating-point-verilog/floating_point_adder.v"

module twosum_step #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I = 1 + EXP_WIDTH_I + MANT_WIDTH_I
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
  // Stage 1: Compute temp = sum + elem
  // ===============================
  logic [BIT_WIDTH_I-1:0] temp_comb;
  logic temp_underflow_flag, temp_overflow_flag, temp_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_temp (
  //   .a(sum_i),
  //   .b(elem_i),
  //   .subtract(1'b0),
  //   .out(temp_comb),
  //   .underflow_flag(temp_underflow_flag),
  //   .overflow_flag(temp_overflow_flag),
  //   .invalid_operation_flag(temp_invalid_operation_flag)
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
  // Stage 2: Compute a_prime = temp - elem
  // ===============================
  logic [BIT_WIDTH_I-1:0] a_prime_comb;
  logic ap_underflow_flag, ap_overflow_flag, ap_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_a_prime (
  //   .a(temp_r),
  //   .b(elem_i),
  //   .subtract(1'b1),
  //   .out(a_prime_comb),
  //   .underflow_flag(ap_underflow_flag),
  //   .overflow_flag(ap_overflow_flag),
  //   .invalid_operation_flag(ap_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign a_prime_comb = temp_r - elem_i;

  logic [BIT_WIDTH_I-1:0] a_prime_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      a_prime_r <= '0;
    else
      a_prime_r <= a_prime_comb;
  end

  // ===============================
  // Stage 3: Compute b_prime = temp - a_prime
  // ===============================
  logic [BIT_WIDTH_I-1:0] b_prime_comb;
  logic bp_underflow_flag, bp_overflow_flag, bp_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_b_prime (
  //   .a(temp_r),
  //   .b(a_prime_r),
  //   .subtract(1'b1),
  //   .out(b_prime_comb),
  //   .underflow_flag(bp_underflow_flag),
  //   .overflow_flag(bp_overflow_flag),
  //   .invalid_operation_flag(bp_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign b_prime_comb = temp_r - a_prime_r;

  logic [BIT_WIDTH_I-1:0] b_prime_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      b_prime_r <= '0;
    else
      b_prime_r <= b_prime_comb;
  end

  // ===============================
  // Stage 4: Compute delta_a = sum - a_prime
  // ===============================
  logic [BIT_WIDTH_I-1:0] delta_a_comb;
  logic da_underflow_flag, da_overflow_flag, da_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_delta_a (
  //   .a(sum_i),
  //   .b(a_prime_r),
  //   .subtract(1'b1),
  //   .out(delta_a_comb),
  //   .underflow_flag(da_underflow_flag),
  //   .overflow_flag(da_overflow_flag),
  //   .invalid_operation_flag(da_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign delta_a_comb = sum_i - a_prime_r;

  logic [BIT_WIDTH_I-1:0] delta_a_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      delta_a_r <= '0;
    else
      delta_a_r <= delta_a_comb;
  end

  // ===============================
  // Stage 5: Compute delta_b = elem - b_prime
  // ===============================
  logic [BIT_WIDTH_I-1:0] delta_b_comb;
  logic db_underflow_flag, db_overflow_flag, db_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_delta_b (
  //   .a(elem_i),
  //   .b(b_prime_r),
  //   .subtract(1'b1),
  //   .out(delta_b_comb),
  //   .underflow_flag(db_underflow_flag),
  //   .overflow_flag(db_overflow_flag),
  //   .invalid_operation_flag(db_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign delta_b_comb = elem_i - b_prime_r;

  logic [BIT_WIDTH_I-1:0] delta_b_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      delta_b_r <= '0;
    else
      delta_b_r <= delta_b_comb;
  end

  // ===============================
  // Stage 6: Compute delta_sum = delta_a + delta_b
  // ===============================
  logic [BIT_WIDTH_I-1:0] delta_sum_comb;
  logic ds_underflow_flag, ds_overflow_flag, ds_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_delta_sum (
  //   .a(delta_a_r),
  //   .b(delta_b_r),
  //   .subtract(1'b0),
  //   .out(delta_sum_comb),
  //   .underflow_flag(ds_underflow_flag),
  //   .overflow_flag(ds_overflow_flag),
  //   .invalid_operation_flag(ds_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign delta_sum_comb = delta_a_r + delta_b_r;

  logic [BIT_WIDTH_I-1:0] delta_sum_r;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      delta_sum_r <= '0;
    else
      delta_sum_r <= delta_sum_comb;
  end

  // ===============================
  // Stage 7: Compute error_o = error_i + delta_sum
  // ===============================
  logic [BIT_WIDTH_I-1:0] error_comb;
  logic e_underflow_flag, e_overflow_flag, e_invalid_operation_flag;

  // floating_point_adder #(
  //   EXP_WIDTH_I, MANT_WIDTH_I
  // ) fp_adder_error_o (
  //   .a(error_i),
  //   .b(delta_sum_r),
  //   .subtract(1'b0),
  //   .out(error_comb),
  //   .underflow_flag(e_underflow_flag),
  //   .overflow_flag(e_overflow_flag),
  //   .invalid_operation_flag(e_invalid_operation_flag)
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign error_comb = error_i + delta_sum_r;

  // ===============================
  // Final Stage: Register outputs
  // ===============================
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_o   <= '0;
      error_o <= '0;
    end else begin
      sum_o   <= temp_r;
      error_o <= error_comb;
    end
  end

endmodule

`endif
