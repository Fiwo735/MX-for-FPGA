`ifndef __NAIVE_ADDER_TREE_SV__
`define __NAIVE_ADDER_TREE_SV__

`include "../parametrizable-floating-point-verilog/floating_point_adder.v"

module naive_adder_tree #(
    parameter  EXP_WIDTH_I  = 8,
    parameter  MANT_WIDTH_I = 23,
    parameter  ELEMS_COUNT  = 32,
    localparam BIT_WIDTH_I  = 1 + EXP_WIDTH_I + MANT_WIDTH_I, // 1 for sign bit
    localparam SUM_WIDTH_O  = BIT_WIDTH_I + $clog2(ELEMS_COUNT),
    localparam TREE_DEPTH   = $clog2(ELEMS_COUNT)
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic signed [BIT_WIDTH_I-1:0] i_vec [ELEMS_COUNT],
    output logic signed [SUM_WIDTH_O-1:0] o_sum
);

    logic signed [BIT_WIDTH_I-1:0] i_vec_reg [ELEMS_COUNT];

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            for (int k = 0; k < ELEMS_COUNT; k++) begin
                i_vec_reg[k] <= '0;  // Reset each element to 0
            end
        end else begin
            i_vec_reg <= i_vec;
        end
    end

    // Define adder tree.
    for(genvar i=0; i<TREE_DEPTH; i++) begin : top_tree
        // Declare adders.
        logic signed [BIT_WIDTH_I+i-1:0] p0_add0 [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i-1:0] p0_add1 [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i:0]   p0_sum  [ELEMS_COUNT>>(1+i)];
        logic underflow_flag [ELEMS_COUNT>>(1+i)];
        logic overflow_flag [ELEMS_COUNT>>(1+i)];
        logic invalid_operation_flag [ELEMS_COUNT>>(1+i)];

        for(genvar j=0; j<ELEMS_COUNT>>(1+i); j++) begin : naive_adders
            floating_point_adder #(
                EXP_WIDTH_I, MANT_WIDTH_I
            ) fp_adder (
                .a(p0_add0[j]),
                .b(p0_add1[j]),
                .subtract(1'b0),

                .out(p0_sum[j]),
                .underflow_flag(underflow_flag[j]),
                .overflow_flag(overflow_flag[j]),
                .invalid_operation_flag(invalid_operation_flag[j])
            );
        end

        // Connections to previous layers.
        if(i != 0) begin
            for(genvar j=0; j<(ELEMS_COUNT>>(1+i)); j++) begin : naive_connections
                always_ff @(posedge clk_i or negedge rst_ni) begin
                    if (!rst_ni) begin
                        p0_add0[j] <= '0;
                        p0_add1[j] <= '0;
                    end else begin
                        p0_add0[j] <= top_tree[i-1].p0_sum[2*j];
                        p0_add1[j] <= top_tree[i-1].p0_sum[2*j+1];
                    end
                end
            end
        end else begin
            for(genvar j=0; j<(ELEMS_COUNT>>(1+i)); j++) begin : naive_starts
                always_ff @(posedge clk_i or negedge rst_ni) begin
                    if (!rst_ni) begin
                        p0_add0[j] <= '0;
                        p0_add1[j] <= '0;
                    end else begin
                        p0_add0[j] <= i_vec_reg[2*j];
                        p0_add1[j] <= i_vec_reg[2*j+1];
                    end
                end
            end
        end
    end

    // Assign outputs.
    assign o_sum = top_tree[TREE_DEPTH-1].p0_sum[0];

endmodule

`endif