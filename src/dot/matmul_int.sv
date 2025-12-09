`ifndef __MATMUL_INT_SV__
`define __MATMUL_INT_SV__

`include "dot_general_int.sv"

module matmul_int #(
    parameter x_rows = 4,
    parameter vec_elem_count = 256, // = x_cols = y_rows
    parameter y_cols = 2,

    parameter k = 32, // block size
    parameter bit_width = 8,
    parameter out_width = 8,
    parameter scale_width = 8,

    localparam x_cols = vec_elem_count,
    localparam y_rows = vec_elem_count,
    localparam block_count = vec_elem_count/k
)(
    input  logic                        i_clk,
    input  logic signed [bit_width-1:0]   A_i [x_rows][x_cols],
    input  logic signed [bit_width-1:0]   B_i [y_rows][y_cols],
    input  logic      [scale_width-1:0] S_A_i [x_rows][block_count],
    input  logic      [scale_width-1:0] S_B_i [block_count][y_cols],
    output logic        [out_width-1:0]   C_o [x_rows][y_cols],
    output logic      [scale_width-1:0] S_C_o [x_rows][y_cols]
);

    for (genvar i=0; i<x_rows; i++) begin : row_loop
        for (genvar j=0; j<y_cols; j++) begin : col_loop
            dot_general_int #(
                .C(vec_elem_count),
                .k(k),
                .bit_width(bit_width),
                .out_width(out_width)
            ) u_dot_general (
                .i_clk(i_clk),
                .i_X(A_i[i]),
                .i_Y(B_i[:,j]),
                .i_S(S_A_i[i]),
                .i_T(S_B_i[:,j]),
                .o_dp(C_o[i][j]),
                .o_scale(S_C_o[i][j])
            );
        end
    end

endmodule

`endif // __MATMUL_INT_SV__