`ifndef __KAHAN_ADDER_TREE_SV__
`define __KAHAN_ADDER_TREE_SV__

`include "../kahan/kahan_start.sv"
`include "../kahan/kahan_merge.sv"

module kahan_adder_tree #(
    parameter  EXP_WIDTH_I  = 5,
    parameter  MANT_WIDTH_I = 2,
    parameter  ELEMS_COUNT  = 32,
    localparam BIT_WIDTH_I  = 1 + EXP_WIDTH_I + MANT_WIDTH_I, // 1 for sign bit
    parameter SUM_WIDTH_O  = BIT_WIDTH_I + $clog2(ELEMS_COUNT),
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
    // Define adder tree.
    generate
    if (ELEMS_COUNT == 2) begin : leaf_node
        logic signed [BIT_WIDTH_I-1:0] sum_res;
        logic signed [BIT_WIDTH_I-1:0] c_res;
        
        // Just one step for 2 elements
        kahan_step #(
            .EXP_WIDTH_I(EXP_WIDTH_I),
            .MANT_WIDTH_I(MANT_WIDTH_I)
        ) kahan_step_leaf (
            .clk_i(clk_i),
            .rst_ni(rst_ni),
            .elem_i(i_vec_reg[0]),
            .c_i('0), // Initial carry is 0
            .sum_i(i_vec_reg[1]),
            .c_o(c_res),
            .sum_o(sum_res)
        );

        always_ff @(posedge clk_i or negedge rst_ni) begin
             if (!rst_ni) o_sum <= '0;
             else o_sum <=  {{(SUM_WIDTH_O - BIT_WIDTH_I){sum_res[BIT_WIDTH_I-1]}}, sum_res}; // Sign Extend
        end

    end else begin : top_tree_gen
        for(genvar i=0; i<TREE_DEPTH; i++) begin : top_tree
            // Declare adders.
            // [FIXED] Removed -1 from width declarations
            logic signed [BIT_WIDTH_I+i:0] sum_a   [ELEMS_COUNT>>(1+i)];
            logic signed [BIT_WIDTH_I+i:0] c_a     [ELEMS_COUNT>>(1+i)];
            logic signed [BIT_WIDTH_I+i:0] sum_b   [ELEMS_COUNT>>(1+i)];
            logic signed [BIT_WIDTH_I+i:0] c_b     [ELEMS_COUNT>>(1+i)];

            logic signed [BIT_WIDTH_I+i:0] sum_res [ELEMS_COUNT>>(1+i)];
            logic signed [BIT_WIDTH_I+i:0] c_res   [ELEMS_COUNT>>(1+i)];

            for(genvar j=0; j<ELEMS_COUNT>>(1+i); j++) begin : kahan_adders
                kahan_merge #(
                    .EXP_WIDTH_I(EXP_WIDTH_I+i),
                    .MANT_WIDTH_I(MANT_WIDTH_I)
                ) kahan_merge_inst (
                    .clk_i(clk_i),
                    .rst_ni(rst_ni),
                    .sum_a_i(sum_a[j]),
                    .c_a_i(c_a[j]),
                    .sum_b_i(sum_b[j]),
                    .c_b_i(c_b[j]),
                    .sum_o(sum_res[j]),
                    .c_o(c_res[j])
                );
            end

            // Connections to previous layers.
            if(i != 0) begin
                for(genvar j=0; j<(ELEMS_COUNT>>(1+i)); j++) begin : kahan_connections
                    always_ff @(posedge clk_i or negedge rst_ni) begin
                        if (!rst_ni) begin
                            sum_a[j] <= '0;
                            c_a[j] <= '0;
                            sum_b[j] <= '0;
                            c_b[j] <= '0;
                        end else begin
                            sum_a[j] <= top_tree[i-1].sum_res[2*j];
                            c_a[j] <= top_tree[i-1].c_res[2*j];
                            sum_b[j] <= top_tree[i-1].sum_res[2*j+1];
                            c_b[j] <= top_tree[i-1].c_res[2*j+1];
                        end
                    end
                end
            end else begin
                for(genvar j=0; j<(ELEMS_COUNT>>2); j++) begin : kahan_starts
                    kahan_start #(
                        .EXP_WIDTH_I(EXP_WIDTH_I+i),
                        .MANT_WIDTH_I(MANT_WIDTH_I)
                    ) kahan_start_inst (
                        .clk_i(clk_i),
                        .rst_ni(rst_ni),
                        .e0(i_vec_reg[4*j]),
                        .e1(i_vec_reg[4*j+1]),
                        .e2(i_vec_reg[4*j+2]),
                        .e3(i_vec_reg[4*j+3]),
                        .sum_a_o(sum_a[j]),
                        .sum_b_o(sum_b[j]),
                        .c_a_o(c_a[j]),
                        .c_b_o(c_b[j])
                    );
                end
            end
        end

        // Assign output
        always_ff @(posedge clk_i or negedge rst_ni) begin
            if (!rst_ni) begin
                o_sum <= '0;
            end else begin
                // [FIXED] removed problematic slice, implicit cast or explicit padding
                o_sum <= top_tree[TREE_DEPTH-1].sum_res[0]; 
            end
        end
    end
    endgenerate

endmodule

`endif