`ifndef __MATMUL_FP_SV__
`define __MATMUL_FP_SV__

// Note: Compiled via explicit read_verilog in run_synth.tcl

// A: [x_rows x x_cols]
// B: [y_rows x y_cols]
// C: [x_rows x y_cols]
// x_cols == y_rows

module matmul_fp #(
    parameter x_rows = 4,
    parameter vec_elem_count = 8, // = x_cols = y_rows
    parameter y_cols = 2,

    parameter k = 2, // block size
    parameter bit_width = 8,
    parameter exp_width = 5, // Defaults to 0 for Integer compatibility
    parameter man_width = 2,
    parameter out_width = 32,
    parameter scale_width = 8,
    parameter string USE_DSP = "auto",

    localparam x_cols = vec_elem_count,
    localparam y_rows = vec_elem_count,
    localparam block_count = vec_elem_count/k
)(
    input  logic                        i_clk,
    input  logic signed   [bit_width-1:0]   A_i [x_rows][x_cols],
    input  logic signed   [bit_width-1:0]   B_i [y_rows][y_cols],
    input  logic        [scale_width-1:0] S_A_i [x_rows][block_count],
    input  logic        [scale_width-1:0] S_B_i [block_count][y_cols],
    output logic signed   [out_width-1:0]   C_o [x_rows][y_cols],
    output logic        [scale_width-1:0] S_C_o [x_rows][y_cols]
);

    // Infer exp_width if not provided

    for (genvar i = 0; i < x_rows; i++) begin : row_loop
        for (genvar j = 0; j < y_cols; j++) begin : col_loop

            logic signed [bit_width-1:0]   B_col [y_rows];
            logic      [scale_width-1:0] S_B_col [block_count];

            for (genvar r = 0; r < y_rows; r++) begin
                assign B_col[r] = B_i[r][j];
            end

            for (genvar r = 0; r < block_count; r++) begin
                assign S_B_col[r] = S_B_i[r][j];
            end

            dot_general_fp #(
                .C(vec_elem_count),
                .k(k),
                .bit_width(bit_width),
                .exp_width(exp_width),
                .man_width(man_width),
                .out_width(out_width),
                .scale_width(scale_width),
                .USE_DSP(USE_DSP)
            ) u_dot_general (
                .i_clk(i_clk),
                .i_X(A_i[i]),
                .i_Y(B_col),
                .i_S(S_A_i[i]),
                .i_T(S_B_col),
                .o_dp(C_o[i][j]),
                .o_scale(S_C_o[i][j])
            );
        end
    end

endmodule

`endif // __MATMUL_FP_SV__
