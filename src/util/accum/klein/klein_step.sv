`ifndef __KLEIN_STEP_SV__
`define __KLEIN_STEP_SV__

`include "../parametrizable-floating-point-verilog/floating_point_adder.v"
`include "../common/abs_bigger_equal.sv"

module klein_step #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I
)(
  input                          clk_i,
  input                          rst_ni,
  input  logic [BIT_WIDTH_I-1:0] elem_i,
  input  logic [BIT_WIDTH_I-1:0] cs_i,
  input  logic [BIT_WIDTH_I-1:0] ccs_i,
  input  logic [BIT_WIDTH_I-1:0] sum_i,
  output logic [BIT_WIDTH_I-1:0] cs_o,
  output logic [BIT_WIDTH_I-1:0] ccs_o,
  output logic [BIT_WIDTH_I-1:0] sum_o
);

  // =========================
  // Stage 1: temp = sum + elem
  // =========================
  logic [BIT_WIDTH_I-1:0] temp_comb;
  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_temp (
  //   .a(sum_i), .b(elem_i), .subtract(1'b0),
  //   .out(temp_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign temp_comb = sum_i + elem_i;

  logic [BIT_WIDTH_I-1:0] temp_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) temp_r <= '0;
    else         temp_r <= temp_comb;

  // =========================
  // Stage 2: Compute sum_minus_temp and elem_minus_temp
  // =========================
  logic [BIT_WIDTH_I-1:0] sum_minus_temp_comb, elem_minus_temp_comb;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_sum_minus_temp (
  //   .a(sum_i), .b(temp_r), .subtract(1'b1), .out(sum_minus_temp_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign sum_minus_temp_comb = sum_i - temp_r;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_elem_minus_temp (
  //   .a(elem_i), .b(temp_r), .subtract(1'b1), .out(elem_minus_temp_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign elem_minus_temp_comb = elem_i - temp_r;

  logic [BIT_WIDTH_I-1:0] sum_minus_temp_r, elem_minus_temp_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) begin
      sum_minus_temp_r <= '0;
      elem_minus_temp_r <= '0;
    end else begin
      sum_minus_temp_r <= sum_minus_temp_comb;
      elem_minus_temp_r <= elem_minus_temp_comb;
    end

  // =========================
  // Stage 3: Compute c_sum1 and c_sum2
  // =========================
  logic [BIT_WIDTH_I-1:0] c_sum1_comb, c_sum2_comb;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_c_sum1 (
  //   .a(sum_minus_temp_r), .b(elem_i), .subtract(1'b0), .out(c_sum1_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign c_sum1_comb = sum_minus_temp_r + elem_i;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_c_sum2 (
  //   .a(elem_minus_temp_r), .b(sum_i), .subtract(1'b0), .out(c_sum2_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign c_sum2_comb = elem_minus_temp_r + sum_i;

  logic [BIT_WIDTH_I-1:0] c_sum1_r, c_sum2_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) begin
      c_sum1_r <= '0;
      c_sum2_r <= '0;
    end else begin
      c_sum1_r <= c_sum1_comb;
      c_sum2_r <= c_sum2_comb;
    end

  // =========================
  // Stage 4: Select c using abs_bigger_equal
  // =========================
  logic decision1;
  abs_bigger_equal #(EXP_WIDTH_I, MANT_WIDTH_I) abs_bigger_equal_inst (
    .a_i(sum_i), .b_i(elem_i), .res_o(decision1)
  );

  logic [BIT_WIDTH_I-1:0] c_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) c_r <= '0;
    else         c_r <= (decision1 ? c_sum1_r : c_sum2_r);

  // =========================
  // Stage 5: temp2 = cs + c
  // =========================
  logic [BIT_WIDTH_I-1:0] temp2_comb;
  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_temp2 (
  //   .a(cs_i), .b(c_r), .subtract(1'b0), .out(temp2_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign temp2_comb = cs_i + c_r;

  logic [BIT_WIDTH_I-1:0] temp2_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) temp2_r <= '0;
    else         temp2_r <= temp2_comb;

  // =========================
  // Stage 6: cs_minus_temp2 and c_minus_temp2
  // =========================
  logic [BIT_WIDTH_I-1:0] cs_minus_temp2_comb, c_minus_temp2_comb;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_cs_minus_temp2 (
  //   .a(cs_i), .b(temp2_r), .subtract(1'b1), .out(cs_minus_temp2_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign cs_minus_temp2_comb = cs_i - temp2_r;
  

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_c_minus_temp2 (
  //   .a(c_r), .b(temp2_r), .subtract(1'b1), .out(c_minus_temp2_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign c_minus_temp2_comb = c_r - temp2_r;

  logic [BIT_WIDTH_I-1:0] cs_minus_temp2_r, c_minus_temp2_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) begin
      cs_minus_temp2_r <= '0;
      c_minus_temp2_r <= '0;
    end else begin
      cs_minus_temp2_r <= cs_minus_temp2_comb;
      c_minus_temp2_r <= c_minus_temp2_comb;
    end

  // =========================
  // Stage 7: cc_sum1 and cc_sum2
  // =========================
  logic [BIT_WIDTH_I-1:0] cc_sum1_comb, cc_sum2_comb;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_cc_sum1 (
  //   .a(cs_minus_temp2_r), .b(c_r), .subtract(1'b0), .out(cc_sum1_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign cc_sum1_comb = cs_minus_temp2_r + c_r;

  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_cc_sum2 (
  //   .a(c_minus_temp2_r), .b(cs_i), .subtract(1'b0), .out(cc_sum2_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  assign cc_sum2_comb = c_minus_temp2_r + cs_i;

  logic [BIT_WIDTH_I-1:0] cc_sum1_r, cc_sum2_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) begin
      cc_sum1_r <= '0;
      cc_sum2_r <= '0;
    end else begin
      cc_sum1_r <= cc_sum1_comb;
      cc_sum2_r <= cc_sum2_comb;
    end

  // =========================
  // Stage 8: Select cc and compute ccs
  // =========================
  logic decision2;
  abs_bigger_equal #(EXP_WIDTH_I, MANT_WIDTH_I) abs_bigger_equal_inst2 (
    .a_i(cs_i), .b_i(c_r), .res_o(decision2)
  );

  logic [BIT_WIDTH_I-1:0] cc_r;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) cc_r <= '0;
    else         cc_r <= (decision2 ? cc_sum1_r : cc_sum2_r);

  logic [BIT_WIDTH_I-1:0] ccs_comb;
  // floating_point_adder #(EXP_WIDTH_I, MANT_WIDTH_I) fp_adder_ccs (
  //   .a(ccs_i), .b(cc_r), .subtract(1'b0), .out(ccs_comb),
  //   .underflow_flag(), .overflow_flag(), .invalid_operation_flag()
  // );
  // Use + or - operator instead of floating_point_adder for better synthesis results
  assign ccs_comb = ccs_i + cc_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cs_o   <= '0;
      ccs_o  <= '0;
      sum_o  <= '0;
    end else begin
      cs_o   <= temp2_r;
      ccs_o  <= ccs_comb;
      sum_o  <= temp_r;
    end
  end

endmodule

`endif
