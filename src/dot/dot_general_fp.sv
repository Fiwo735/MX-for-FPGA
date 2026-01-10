`ifndef __DOT_GENERAL_FP_SV__
`define __DOT_GENERAL_FP_SV__

// Note: Compiled via explicit read_verilog in run_synth.tcl

// Handles tree reduction of specific FP blocks reusing add_nrm.

module dot_general_fp #(
    parameter C = 8,
    parameter k = 4,
    parameter bit_width = 8,
    parameter exp_width = 5,  // Extra param for FP
    parameter man_width = 2,  // Extra param for FP
    parameter out_width = 8,
    parameter scale_width = 8, // Added
    parameter string USE_DSP = "auto",
    parameter string ACCUM_METHOD = "Kulisch",

    localparam dp_width = 2*bit_width + $clog2(k), 
    // Note: dp_width might need tuning for FP accumulation size, 
    // but using roughly standard expansion is safe for now.
    
    localparam block_count = C/k,
    localparam tree_depth = $clog2(block_count)
)(
    input  logic i_clk,
    input  logic signed [bit_width-1:0] i_X     [C],
    input  logic signed [bit_width-1:0] i_Y     [C],
    input  logic             [scale_width-1:0] i_S     [block_count],
    input  logic             [scale_width-1:0] i_T     [block_count],
    output logic        [out_width-1:0] o_dp,
    output logic             [scale_width-1:0] o_scale
);

    // Sum within blocks
    logic signed [dp_width-1:0] dot_out [block_count];

    for(genvar i=0; i<(block_count); i++) begin
        dot_fp #(
            .exp_width(exp_width),
            .man_width(man_width),
            .k(k),
            // We override bit_width derived parameters if needed, 
            // but dot_fp derives them from exp/man/k. A mismatch check might be good.
            .bit_width(bit_width),
            .out_width(dp_width),
            .USE_DSP(USE_DSP),
            .ACCUM_METHOD(ACCUM_METHOD)
        ) u_dot_fp (
            .i_vec_a(i_X[i*k +: k]),
            .i_vec_b(i_Y[i*k +: k]),
            .o_dp(dot_out[i])
        );
    end

    // Sum scales
    logic [scale_width-1:0] dot_scales [block_count];

    for(genvar i=0; i<(block_count); i++) begin
        assign dot_scales[i] = i_S[i] + i_T[i];
    end

    // Sum across blocks (Tree Reduction)
    // We reuse the existing tree structure from dot_general_int
    // because add_nrm is generic enough to handle (Value, Scale) + (Value, Scale)
    // regardless of whether Value came from Int or FP dot product.
    
    for(genvar i=0; i<tree_depth; i++) begin : tree_add
        // Declare adders.
        logic signed [dp_width-1:0] p0_add0   [block_count>>(1+i)];
        logic signed [dp_width-1:0] p0_add1   [block_count>>(1+i)];
        logic signed [dp_width-1:0] p0_sum    [block_count>>(1+i)];
        logic signed     [scale_width-1:0] p0_scale0 [block_count>>(1+i)];
        logic signed     [scale_width-1:0] p0_scale1 [block_count>>(1+i)];
        logic signed     [scale_width-1:0] p0_scale  [block_count>>(1+i)];

        for(genvar j=0; j<block_count>>(1+i); j++) begin
            add_nrm #(
                .int_w(dp_width)
            ) u_add_nrm (
                .i_op0(p0_add0[j]),
                .i_op1(p0_add1[j]),
                .i_scale0(p0_scale0[j]),
                .i_scale1(p0_scale1[j]),
                .out(p0_sum[j]),
                .o_scale(p0_scale[j])
            );
        end

        // Connections to previous layers.
        if(i != 0) begin
            for(genvar j=0; j<(block_count>>(1+i)); j++) begin
                assign p0_add0[j] = tree_add[i-1].p0_sum[2*j];
                assign p0_add1[j] = tree_add[i-1].p0_sum[2*j+1];
                assign p0_scale0[j] = tree_add[i-1].p0_scale[2*j];
                assign p0_scale1[j] = tree_add[i-1].p0_scale[2*j+1];
            end
        end else begin
            for(genvar j=0; j<(block_count>>(1+i)); j++) begin
                assign p0_add0[j] = dot_out[2*j];
                assign p0_add1[j] = dot_out[2*j+1];
                assign p0_scale0[j] = dot_scales[2*j];
                assign p0_scale1[j] = dot_scales[2*j+1];
            end
        end
    end


    // Form output
    assign o_dp = tree_add[tree_depth-1].p0_sum[0];
    assign o_scale = tree_add[tree_depth-1].p0_scale[0];

endmodule

`endif // __DOT_GENERAL_FP_SV__
