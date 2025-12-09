`ifndef __DOT_GENERAL_INT_SV__
`define __DOT_GENERAL_INT_SV__

`include "../../dot/dot_int.sv"
`include "../../util/arith/add_nrm.sv"

module dot_general_int #(
    parameter C = 8,
    parameter k = 4,
    parameter bit_width = 8,
    parameter out_width = 8,

    localparam dp_width = 2*bit_width + $clog2(k),
    localparam block_count = C/k,
    localparam tree_depth = $clog2(block_count)
)(
    input  logic i_clk,
    input  logic signed [bit_width-1:0] i_X     [C],
    input  logic signed [bit_width-1:0] i_Y     [C],
    input  logic                [8-1:0] i_S     [block_count],
    input  logic                [8-1:0] i_T     [block_count],
    output logic        [out_width-1:0] o_dp,
    output logic                [8-1:0] o_scale
);


    // Sum within blocks
    logic signed [dp_width-1:0] dot_out [block_count];

    for(genvar i=0; i<(block_count); i++) begin
        dot_int #(
            .bit_width(bit_width),
            .k(k)
        ) u_dot_int (
            .i_vec_a(i_X[i*k +: k]),
            .i_vec_b(i_Y[i*k +: k]),
            .o_dp(dot_out[i])
        );
    end

    // Sum scales
    logic [8-1:0] dot_scales [block_count];

    for(genvar i=0; i<(block_count); i++) begin
        assign dot_scales[i] = i_S[i] + i_T[i];// - 8'd127; ?
    end

    // Sum across blocks
    for(genvar i=0; i<tree_depth; i++) begin : tree_add
        // Declare adders.
        logic signed [dp_width-1:0] p0_add0   [block_count>>(1+i)];
        logic signed [dp_width-1:0] p0_add1   [block_count>>(1+i)];
        logic signed [dp_width-1:0] p0_sum    [block_count>>(1+i)];
        logic signed        [8-1:0] p0_scale0 [block_count>>(1+i)];
        logic signed        [8-1:0] p0_scale1 [block_count>>(1+i)];
        logic signed        [8-1:0] p0_scale  [block_count>>(1+i)];

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

`endif // __DOT_GENERAL_INT_SV__