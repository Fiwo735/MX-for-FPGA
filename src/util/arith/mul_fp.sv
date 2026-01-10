`ifndef __MUL_FP_SV__
`define __MUL_FP_SV__

`include "../util/arith/mul_int.sv"

module mul_fp #(
    parameter exp_width = 5,
    parameter man_width = 2,
    parameter bit_width = 1 + exp_width + man_width,
    parameter fi_width = man_width + 2,
    parameter fi_prd_width = 2 * fi_width,
    parameter prd_width = 2 * ((1<<exp_width) + man_width),
    parameter string USE_DSP = "auto"
)(
    input  logic signed [bit_width-1:0] i_op0,
    input  logic signed [bit_width-1:0] i_op1,
    output logic signed [prd_width-1:0] o_prd
);

    generate
        if (exp_width > 0) begin : gen_fp_mult
            logic                 op0_sgn;
            logic                 op1_sgn;
            logic [exp_width-1:0] op0_exp;
            logic [exp_width-1:0] op1_exp;
            logic   [man_width:0] op0_man_ext;
            logic   [man_width:0] op1_man_ext;
            logic op0_nrm;
            logic op1_nrm;

            assign op0_sgn = i_op0[bit_width-1];
            assign op1_sgn = i_op1[bit_width-1];
            assign op0_exp = i_op0[bit_width-2:man_width];
            assign op1_exp = i_op1[bit_width-2:man_width];
            assign op0_man_ext = {|op0_exp, i_op0[man_width-1:0]};
            assign op1_man_ext = {|op1_exp, i_op1[man_width-1:0]};
            assign op0_nrm = |op0_exp;
            assign op1_nrm = |op1_exp;

            logic signed [fi_width-1:0] op0_signed_man;
            logic signed [fi_width-1:0] op1_signed_man;
            assign op0_signed_man = op0_sgn ? -op0_man_ext : op0_man_ext;
            assign op1_signed_man = op1_sgn ? -op1_man_ext : op1_man_ext;

            logic signed [fi_prd_width-1:0] prd_fi;
            mul_int #(
                .bit_width(fi_width),
                .USE_DSP(USE_DSP)
            ) u_int_mul (
                .i_op0(op0_signed_man),
                .i_op1(op1_signed_man),
                .o_prd(prd_fi)
            );
            
            logic signed [prd_width-1:0] prd_shifted;
            assign prd_shifted = prd_fi << ($unsigned({1'b0, op0_exp}) + $unsigned({1'b0, op1_exp}) - $unsigned(op0_nrm) - $unsigned(op1_nrm));
            assign o_prd = prd_shifted;

        end else begin : gen_int_mult
             // Integer Mode: Direct Pass-Through
             // i_op0 is assumed to be signed 2's complement
             // We instantiate mul_int directly with correct bit_width
             mul_int #(
                .bit_width(bit_width),
                .USE_DSP(USE_DSP)
             ) u_int_mul (
                .i_op0(i_op0),
                .i_op1(i_op1),
                .o_prd(o_prd)
             );
        end
    endgenerate

endmodule

`endif // __MUL_FP_SV__