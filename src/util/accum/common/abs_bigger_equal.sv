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
  logic [MANT_WIDTH_I-1:0] a_mant;
  logic [MANT_WIDTH_I-1:0] b_mant;
  
  assign a_mant = a_i[MANT_WIDTH_I-1:0];
  assign b_mant = b_i[MANT_WIDTH_I-1:0];

  generate
    if (EXP_WIDTH_I > 0) begin : gen_has_exp
        logic [EXP_WIDTH_I-1:0]  a_exp;
        logic [EXP_WIDTH_I-1:0]  b_exp;
        
        assign a_exp = a_i[BIT_WIDTH_I-2:MANT_WIDTH_I];
        assign b_exp = b_i[BIT_WIDTH_I-2:MANT_WIDTH_I];

        always_comb begin
            if (a_exp > b_exp) begin
                res_o = 1'b1;
            end else if (a_exp < b_exp) begin
                res_o = 1'b0;
            end else begin
                // Exponents equal, compare mantissas
                if (a_mant >= b_mant) res_o = 1'b1;
                else                  res_o = 1'b0;
            end
        end
    end else begin : gen_no_exp
        // Integer mode (No exponent)
        // Just compare magnitude (mantissa)
        always_comb begin
            if (a_mant >= b_mant) res_o = 1'b1;
            else                  res_o = 1'b0;
        end
    end
  endgenerate

endmodule

`endif