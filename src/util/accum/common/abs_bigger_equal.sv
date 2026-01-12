`ifndef __ABS_BIGGER_EQUAL_SV__
`define __ABS_BIGGER_EQUAL_SV__

module abs_bigger_equal #(
    parameter EXP_WIDTH_I  = 5,
    parameter MANT_WIDTH_I = 2,
    localparam BIT_WIDTH_I = 1 + EXP_WIDTH_I + MANT_WIDTH_I // 1 for sign bit
)(
    input  logic signed [BIT_WIDTH_I-1:0] a_i,
    input  logic signed [BIT_WIDTH_I-1:0] b_i,
    output logic                          res_o
);

  // Sign doesn't matter for absolute value comparison
  logic [EXP_WIDTH_I-1:0]  a_exp = a_i[BIT_WIDTH_I-2:MANT_WIDTH_I];
  logic [MANT_WIDTH_I-1:0] a_mant = a_i[MANT_WIDTH_I-1:0];
  logic [EXP_WIDTH_I-1:0]  b_exp = b_i[BIT_WIDTH_I-2:MANT_WIDTH_I];
  logic [MANT_WIDTH_I-1:0] b_mant = b_i[MANT_WIDTH_I-1:0];

  // Compare absolute values of a_i and b_i
  always_comb begin
    if (a_exp >= b_exp) begin
      res_o = 1'b1;
    end else if (a_exp < b_exp) begin
      res_o = 1'b0;
    end else begin // a_exp == b_exp
      if (a_mant >= b_mant) begin
        res_o = 1'b1; // a_i is greater than or equal to b_i
      end else begin
        res_o = 1'b0; // a_i is less than b_i
      end
    end
  end

endmodule

`endif