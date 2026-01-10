`ifndef __DOT_FP_SV__
`define __DOT_FP_SV__

// Note: Compiled via explicit read_verilog in run_synth.tcl

module dot_fp #(
    parameter exp_width = 5,
    parameter man_width = 2,
    parameter k         = 32,
    parameter bit_width = 1 + exp_width + man_width,
    parameter fi_width  = man_width + 2,
    parameter prd_width = 2 * ((1<<exp_width) + man_width),
    parameter out_width = prd_width + $clog2(k),
    parameter string USE_DSP = "auto",
    parameter string ACCUM_METHOD = "Kulisch"
)(
    input  logic signed [bit_width-1:0] i_vec_a [k],
    input  logic signed [bit_width-1:0] i_vec_b [k],
    output logic signed [out_width-1:0] o_dp
);

    // Perform multiplications.
    logic signed [prd_width-1:0] p0_prd [k];

    vec_mul_fp #(
        .exp_width(exp_width),
        .man_width(man_width),
        .length(k),
        .USE_DSP(USE_DSP)
    ) u_vec_mul (
        .i_vec_a(i_vec_a),
        .i_vec_b(i_vec_b),
        .o_prd(p0_prd)
    );

    // Calculate sum.
    logic signed [out_width-1:0] p0_sum;

    generate
        if (ACCUM_METHOD == "KULISCH") begin : gen_kulisch_accum
            vec_sum_int #(
                .bit_width(prd_width),
                .length(k)
            ) u_tree_add (
                .i_vec(p0_prd),
                .o_sum(p0_sum)
            );
        end else begin : gen_error_accum
            $error("Unsupported ACCUM_METHOD in dot_fp:");
        end
    endgenerate

    

    assign o_dp = p0_sum;


endmodule

`endif // __DOT_FP_SV__
