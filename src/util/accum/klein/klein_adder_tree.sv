`ifndef __KLEIN_ADDER_TREE_SV__
`define __KLEIN_ADDER_TREE_SV__

`include "../klein/klein_start.sv"
`include "../klein/klein_merge.sv"
`include "../parametrizable-floating-point-verilog/floating_point_adder.v"

module klein_adder_tree #(
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
    for(genvar i=0; i<TREE_DEPTH; i++) begin : top_tree
        // Declare adders.
        logic signed [BIT_WIDTH_I+i-1:0] sum_a [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i-1:0] sum_b [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i-1:0] cs_a [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i-1:0] cs_b [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i-1:0] ccs_a [ELEMS_COUNT>>(1+i)];
        logic signed [BIT_WIDTH_I+i-1:0] ccs_b [ELEMS_COUNT>>(1+i)];

        logic signed   [BIT_WIDTH_I+i-1:0] sum_res  [ELEMS_COUNT>>(1+i)];
        logic signed   [BIT_WIDTH_I+i-1:0] cs_res  [ELEMS_COUNT>>(1+i)];
        logic signed   [BIT_WIDTH_I+i-1:0] ccs_res  [ELEMS_COUNT>>(1+i)];

        for(genvar j=0; j<ELEMS_COUNT>>(1+i); j++) begin : klein_adders
            klein_merge #(
                .EXP_WIDTH_I(EXP_WIDTH_I+i),
                .MANT_WIDTH_I(MANT_WIDTH_I)
            ) neumaier_merge_inst (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .sum_a_i(sum_a[j]),
                .cs_a_i(cs_a[j]),
                .ccs_a_i(ccs_a[j]),
                .sum_b_i(sum_b[j]),
                .cs_b_i(cs_b[j]),
                .ccs_b_i(ccs_b[j]),
                .sum_o(sum_res[j]),
                .cs_o(cs_res[j]),
                .ccs_o(ccs_res[j])
            );
        end

        // Connections to previous layers.
        if(i != 0) begin
            for(genvar j=0; j<(ELEMS_COUNT>>(1+i)); j++) begin : klein_connections
                always_ff @(posedge clk_i or negedge rst_ni) begin
                    if (!rst_ni) begin
                        sum_a[j] <= '0;
                        cs_a[j] <= '0;
                        ccs_a[j] <= '0;
                        sum_b[j] <= '0;
                        cs_b[j] <= '0;
                        ccs_b[j] <= '0;
                    end else begin
                        sum_a[j] <= top_tree[i-1].sum_res[2*j];
                        cs_a[j] <= top_tree[i-1].cs_res[2*j];
                        ccs_a[j] <= top_tree[i-1].ccs_res[2*j];
                        sum_b[j] <= top_tree[i-1].sum_res[2*j+1];
                        cs_b[j] <= top_tree[i-1].cs_res[2*j+1];
                        ccs_b[j] <= top_tree[i-1].ccs_res[2*j+1];
                    end
                end
            end
        end else begin
            for(genvar j=0; j<(ELEMS_COUNT>>2); j++) begin : klein_starts
                klein_start #(
                    .EXP_WIDTH_I(EXP_WIDTH_I+i),
                    .MANT_WIDTH_I(MANT_WIDTH_I)
                ) neumaier_start_inst (
                    .clk_i(clk_i),
                    .rst_ni(rst_ni),
                    .e0(i_vec_reg[4*j]),
                    .e1(i_vec_reg[4*j+1]),
                    .e2(i_vec_reg[4*j+2]),
                    .e3(i_vec_reg[4*j+3]),
                    .sum_a_o(sum_a[j]),
                    .sum_b_o(sum_b[j]),
                    .cs_a_o(cs_a[j]),
                    .cs_b_o(cs_b[j]),
                    .ccs_a_o(ccs_a[j]),
                    .ccs_b_o(ccs_b[j])
                );
            end
        end
    end

    
    logic signed [BIT_WIDTH_I+TREE_DEPTH-1:0] final_sum_res;
    logic signed [BIT_WIDTH_I+TREE_DEPTH-1:0] final_cs_res;
    logic signed [BIT_WIDTH_I+TREE_DEPTH-1:0] final_ccs_res;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            final_sum_res <= '0;
            final_cs_res <= '0;
            final_ccs_res <= '0;
        end else begin
            final_sum_res <= top_tree[TREE_DEPTH-1].sum_res[0];
            final_cs_res <= top_tree[TREE_DEPTH-1].cs_res[0];
            final_ccs_res <= top_tree[TREE_DEPTH-1].ccs_res[0];
        end
    end

    // Assign output by adding the final results of the tree.
    logic res1_underflow_flag, res1_overflow_flag, res1_invalid_operation_flag;
    logic signed [SUM_WIDTH_O-1:0] res1_sum;
    // floating_point_adder #(
    //     EXP_WIDTH_I + $clog2(ELEMS_COUNT), MANT_WIDTH_I
    // ) fp_adder_res (
    //     .a(final_cs_res),
    //     .b(final_ccs_res),
    //     .subtract(1'b0),

    //     .out(res1_sum),
    //     .underflow_flag(res1_underflow_flag),
    //     .overflow_flag(res1_overflow_flag),
    //     .invalid_operation_flag(res1_invalid_operation_flag)
    // );
    // Use + or - operator instead of floating_point_adder for better synthesis results
    assign res1_sum = final_cs_res + final_ccs_res;

    logic signed [SUM_WIDTH_O-1:0]   o_sum_reg;
    logic res2_underflow_flag, res2_overflow_flag, res2_invalid_operation_flag;
    // floating_point_adder #(
    //     EXP_WIDTH_I + $clog2(ELEMS_COUNT), MANT_WIDTH_I
    // ) fp_adder_res2 (
    //     .a(final_sum_res),
    //     .b(res1_sum),
    //     .subtract(1'b0),
    //     .out(o_sum_reg),
    //     .underflow_flag(res2_underflow_flag),
    //     .overflow_flag(res2_overflow_flag),
    //     .invalid_operation_flag(res2_invalid_operation_flag)
    // );
    // Use + or - operator instead of floating_point_adder for better synthesis results
    assign o_sum_reg = final_sum_res + res1_sum;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            o_sum <= '0;
        end else begin
            o_sum <= o_sum_reg;
        end
    end

endmodule

`endif