`ifndef __SYSTOLIC_MATMUL_INT_SV__
`define __SYSTOLIC_MATMUL_INT_SV__

`include "matmul_int.sv"

module systolic_matmul_int #(
    parameter PE_rows        = 2,
    parameter PE_cols        = 2,
    parameter x_rows         = 4,
    parameter vec_elem_count = 8, // = x_cols = y_rows
    parameter y_cols         = 4,
    parameter k              = 2,
    parameter bit_width      = 8,
    parameter out_width      = 8,
    parameter scale_width    = 8,

    localparam block_count = vec_elem_count / k
)(
    input  logic i_clk,

    // External matrix inputs (clean API)
    input  logic signed   [bit_width-1:0]   A_in [x_rows][vec_elem_count],
    input  logic        [scale_width-1:0] S_A_in [x_rows][block_count],

    input  logic signed   [bit_width-1:0]   B_in [vec_elem_count][y_cols],
    input  logic        [scale_width-1:0] S_B_in [block_count][y_cols],

    // Output matrix C
    output logic signed   [out_width-1:0]   C_out [x_rows][y_cols],
    output logic        [scale_width-1:0] S_C_out [x_rows][y_cols]
);

    // ----------------------------
    // Internal systolic pipelines
    // ----------------------------

    logic signed   [bit_width-1:0]   A_pipe [PE_rows][PE_cols+1][x_rows][vec_elem_count];
    logic        [scale_width-1:0] S_A_pipe [PE_rows][PE_cols+1][x_rows][block_count];

    logic signed   [bit_width-1:0]   B_pipe [PE_rows+1][PE_cols][vec_elem_count][y_cols];
    logic        [scale_width-1:0] S_B_pipe [PE_rows+1][PE_cols][block_count][y_cols];

    // ---------------------------------
    // Reshape & inject A (row-wise)
    // ---------------------------------
    // Replicate A across PE rows, inject into column 0
    for (genvar i = 0; i < PE_rows; i++) begin : init_a
        assign   A_pipe[i][0] =   A_in;
        assign S_A_pipe[i][0] = S_A_in;
    end

    // ---------------------------------
    // Reshape & inject B (column-wise)
    // ---------------------------------
    // Replicate B across PE cols, inject into row 0
    for (genvar j = 0; j < PE_cols; j++) begin : init_b
        assign   B_pipe[0][j] =   B_in;
        assign S_B_pipe[0][j] = S_B_in;
    end

    // ----------------------------
    // PE grid
    // ----------------------------

    // Declare internal wires for each PE output
    logic signed   [out_width-1:0]   C_pe [PE_rows][PE_cols][x_rows][y_cols];
    logic        [scale_width-1:0] S_C_pe [PE_rows][PE_cols][x_rows][y_cols];

    for (genvar i = 0; i < PE_rows; i++) begin : row_loop
        for (genvar j = 0; j < PE_cols; j++) begin : col_loop

            matmul_int #(
                .x_rows(x_rows),
                .vec_elem_count(vec_elem_count),
                .y_cols(y_cols),
                .k(k),
                .bit_width(bit_width),
                .out_width(out_width),
                .scale_width(scale_width)
            ) u_pe (
                .i_clk(i_clk),
                .A_i   (A_pipe[i][j]),
                .B_i   (B_pipe[i][j]),
                .S_A_i (S_A_pipe[i][j]),
                .S_B_i (S_B_pipe[i][j]),
                .C_o   (C_pe[i][j]),
                .S_C_o (S_C_pe[i][j])
            );

            // Forward A east
            assign   A_pipe[i][j+1] =   A_pipe[i][j];
            assign S_A_pipe[i][j+1] = S_A_pipe[i][j];

            // Forward B south
            assign   B_pipe[i+1][j] =   B_pipe[i][j];
            assign S_B_pipe[i+1][j] = S_B_pipe[i][j];

        end
    end

    // ---------------------------------
    // Output selection (bottom-right PE)
    // ---------------------------------
    assign C_out   = C_pe[PE_rows-1][PE_cols-1];
    assign S_C_out = S_C_pe[PE_rows-1][PE_cols-1];

endmodule

`endif // __SYSTOLIC_MATMUL_INT_SV__
