`ifndef __KLEIN_MERGE_SV__
`define __KLEIN_MERGE_SV__

`include "../klein/klein_step.sv"

module klein_merge #(
  parameter  EXP_WIDTH_I =  5,
  parameter  MANT_WIDTH_I = 2,
  localparam BIT_WIDTH_I =  1 + EXP_WIDTH_I + MANT_WIDTH_I
)(
  input                          clk_i,
  input                          rst_ni,

  input  logic [BIT_WIDTH_I-1:0] sum_a_i,
  input  logic [BIT_WIDTH_I-1:0] sum_b_i,
  input  logic [BIT_WIDTH_I-1:0] cs_a_i,
  input  logic [BIT_WIDTH_I-1:0] cs_b_i,
  input  logic [BIT_WIDTH_I-1:0] ccs_a_i,
  input  logic [BIT_WIDTH_I-1:0] ccs_b_i,

  output logic [BIT_WIDTH_I-1:0] sum_o,
  output logic [BIT_WIDTH_I-1:0] cs_o,
  output logic [BIT_WIDTH_I-1:0] ccs_o
);

  // ====================================
  // Stage 0: Register inputs
  // ====================================
  logic [BIT_WIDTH_I-1:0] sum_a_r, sum_b_r, cs_a_r, cs_b_r, ccs_a_r, ccs_b_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_a_r  <= '0;
      sum_b_r  <= '0;
      cs_a_r   <= '0;
      cs_b_r   <= '0;
      ccs_a_r  <= '0;
      ccs_b_r  <= '0;
    end else begin
      sum_a_r  <= sum_a_i;
      sum_b_r  <= sum_b_i;
      cs_a_r   <= cs_a_i;
      cs_b_r   <= cs_b_i;
      ccs_a_r  <= ccs_a_i;
      ccs_b_r  <= ccs_b_i;
    end
  end

  // ====================================
  // Stage 1: klein_step_1
  // ====================================
  logic [BIT_WIDTH_I-1:0] temp_sum_s1, temp_cs_s1, temp_ccs_s1;

  klein_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) klein_step_1 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(sum_a_r),
    .cs_i(cs_a_r),
    .ccs_i(ccs_a_r),
    .sum_i(sum_b_r),
    .cs_o(temp_cs_s1),
    .ccs_o(temp_ccs_s1),
    .sum_o(temp_sum_s1)
  );

  // ====================================
  // Stage 2: Register outputs of step 1
  // ====================================
  logic [BIT_WIDTH_I-1:0] temp_sum_r1, temp_cs_r1, temp_ccs_r1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      temp_sum_r1 <= '0;
      temp_cs_r1  <= '0;
      temp_ccs_r1 <= '0;
    end else begin
      temp_sum_r1 <= temp_sum_s1;
      temp_cs_r1  <= temp_cs_s1;
      temp_ccs_r1 <= temp_ccs_s1;
    end
  end

  // ====================================
  // Stage 3: klein_step_2
  // ====================================
  logic [BIT_WIDTH_I-1:0] temp_sum_s2, temp_cs_s2, temp_ccs_s2;

  klein_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) klein_step_2 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(temp_sum_r1),
    .cs_i(temp_cs_r1),
    .ccs_i(temp_ccs_r1),
    .sum_i(cs_b_r),
    .cs_o(temp_cs_s2),
    .ccs_o(temp_ccs_s2),
    .sum_o(temp_sum_s2)
  );

  // ====================================
  // Stage 4: Register outputs of step 2
  // ====================================
  logic [BIT_WIDTH_I-1:0] temp_sum_r2, temp_cs_r2, temp_ccs_r2;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      temp_sum_r2 <= '0;
      temp_cs_r2  <= '0;
      temp_ccs_r2 <= '0;
    end else begin
      temp_sum_r2 <= temp_sum_s2;
      temp_cs_r2  <= temp_cs_s2;
      temp_ccs_r2 <= temp_ccs_s2;
    end
  end

  // ====================================
  // Stage 5: klein_step_3
  // ====================================
  logic [BIT_WIDTH_I-1:0] sum_s3, cs_s3, ccs_s3;

  klein_step #(
    .EXP_WIDTH_I(EXP_WIDTH_I),
    .MANT_WIDTH_I(MANT_WIDTH_I)
  ) klein_step_3 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .elem_i(temp_sum_r2),
    .cs_i(temp_cs_r2),
    .ccs_i(temp_ccs_r2),
    .sum_i(ccs_b_r),
    .cs_o(cs_s3),
    .ccs_o(ccs_s3),
    .sum_o(sum_s3)
  );

  // ====================================
  // Stage 6: Register final outputs
  // ====================================
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_o  <= '0;
      cs_o   <= '0;
      ccs_o  <= '0;
    end else begin
      sum_o  <= sum_s3;
      cs_o   <= cs_s3;
      ccs_o  <= ccs_s3;
    end
  end

endmodule

`endif
