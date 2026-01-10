`ifndef __MUL_INT_SV__
`define __MUL_INT_SV__

module mul_int #(
    parameter bit_width = 8,
    parameter prd_width = 2*bit_width,
    parameter string USE_DSP = "auto"
)(
    input  logic signed [bit_width-1:0] i_op0,
    input  logic signed [bit_width-1:0] i_op1,
    output logic signed [prd_width-1:0] o_prd
);


    generate
        if (USE_DSP == "yes") begin : g_dsp
            // Complex (Sign/Mag) implementation: Maps reliably to DSP48
            logic unsigned [bit_width-1:0] u_op0;
            logic unsigned [bit_width-1:0] u_op1;
            logic unsigned [prd_width-1:0] u_prd;
            logic prd_sign;

            always_comb begin
                prd_sign = ((i_op0 < 0) ^ (i_op1 < 0));
                u_op0 = (i_op0 < 0) ? -i_op0 : i_op0;
                u_op1 = (i_op1 < 0) ? -i_op1 : i_op1;
                (* use_dsp = "yes" *) u_prd = u_op0 * u_op1;
                o_prd = prd_sign ? -u_prd : u_prd;
            end
        end else if (USE_DSP == "logic") begin : g_logic
            // Simple implementation with forced logic attribute
            logic signed [prd_width-1:0] product;
            always_comb begin
                (* use_dsp = "logic" *) product = i_op0 * i_op1;
            end
            assign o_prd = product;
        end else begin : g_auto
            // Default/Auto implementation: No attributes, let Vivado decide
            logic signed [prd_width-1:0] product;
            always_comb product = i_op0 * i_op1;
            assign o_prd = product;
        end
    endgenerate

endmodule

`endif // __MUL_INT_SV__