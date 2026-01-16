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
    parameter string ACCUM_METHOD = "KULISCH"
)(
    input logic i_clk,  
    input  logic signed [bit_width-1:0] i_vec_a [k],
    input  logic signed [bit_width-1:0] i_vec_b [k],
    output logic signed [out_width-1:0] o_dp
);

    // Elementwise multiplication of vectors.
    logic signed [prd_width-1:0] p0_prd_comb [k];
    logic signed [prd_width-1:0] p0_prd [k];
    
    always_ff @(posedge i_clk) begin
        for(int i=0; i<k; i++) begin
             p0_prd[i] <= p0_prd_comb[i];
        end
    end

    vec_mul_fp #(
        .exp_width(exp_width),
        .man_width(man_width),
        .length(k),
        .USE_DSP(USE_DSP)
    ) u_vec_mul (
        .i_vec_a(i_vec_a),
        .i_vec_b(i_vec_b),
        .o_prd(p0_prd_comb)
    );

    // Calculate sum.
    logic signed [out_width-1:0] p0_sum;

    generate
        if (ACCUM_METHOD == "KULISCH") begin : gen_kulisch_accum
            vec_sum_int #(
                .bit_width(prd_width),
                .length(k),
                .sum_width(out_width) // Enforce output width
            ) u_tree_add (
                .i_clk(i_clk),
                .i_vec(p0_prd),
                .o_sum(p0_sum)
            );
        end else begin : gen_other_accum
            // Truncate p0_prd to bit_width for other accumulation methods.
            logic signed [bit_width-1:0] p0_prd_trunc [k];
            for (genvar i=0; i<k; i++) begin
                assign p0_prd_trunc[i] = p0_prd[i][bit_width-1:0];
            end

            // Use other accumulation methods
            if (ACCUM_METHOD == "KAHAN") begin : gen_kahan_accum
                kahan_adder_tree #(
                    .EXP_WIDTH_I(exp_width),
                    .MANT_WIDTH_I(man_width),
                    .ELEMS_COUNT(k),
                    .SUM_WIDTH_O(out_width)
                ) u_kahan_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(p0_prd_trunc),
                    .o_sum(p0_sum)
                );
            end else if (ACCUM_METHOD == "TWOSUM") begin : gen_twosum_accum
                twosum_adder_tree #(
                    .EXP_WIDTH_I(exp_width),
                    .MANT_WIDTH_I(man_width),
                    .ELEMS_COUNT(k),
                    .SUM_WIDTH_O(out_width)
                ) u_twosum_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(p0_prd_trunc),
                    .o_sum(p0_sum)
                );
            end else if (ACCUM_METHOD == "FASTTWOSUM") begin : gen_fasttwosum_accum
                fasttwosum_adder_tree #(
                    .EXP_WIDTH_I(exp_width),
                    .MANT_WIDTH_I(man_width),
                    .ELEMS_COUNT(k),
                    .SUM_WIDTH_O(out_width)
                ) u_fasttwosum_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(p0_prd_trunc),
                    .o_sum(p0_sum)
                );
            end else if (ACCUM_METHOD == "NEUMAIER") begin : gen_neumaier_accum
                neumaier_adder_tree #(
                    .EXP_WIDTH_I(exp_width),
                    .MANT_WIDTH_I(man_width),
                    .ELEMS_COUNT(k),
                    .SUM_WIDTH_O(out_width)
                ) u_neumaier_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(p0_prd_trunc),
                    .o_sum(p0_sum)
                );
            end else if (ACCUM_METHOD == "KLEIN") begin : gen_klein_accum
                klein_adder_tree #(
                    .EXP_WIDTH_I(exp_width),
                    .MANT_WIDTH_I(man_width),
                    .ELEMS_COUNT(k),
                    .SUM_WIDTH_O(out_width)
                ) u_klein_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(p0_prd_trunc),
                    .o_sum(p0_sum)
                );
            end else begin : gen_error_accum
                $error("Unsupported ACCUM_METHOD");
            end
        end
    endgenerate

    

    assign o_dp = p0_sum;


endmodule

`endif // __DOT_FP_SV__
