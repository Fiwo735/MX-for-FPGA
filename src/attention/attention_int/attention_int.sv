`ifndef __ATTENTION_INT_SV__
`define __ATTENTION_INT_SV__

`include "../../matmul/matmul_int/matmul_int.sv"

// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_kq)) * V
// Q: [S_q x d_kq]
// K: [S_kv x d_kq]
// V: [S_kv x d_v]
// K^T: [d_kq x S_kv]
// Q * K^T: [S_q x S_kv]
// Output: [S_q x d_v]

// S_x are typically larger than d_x
// Values along d_x dimension belong to one embedding,
// hence it's better to group into MX blocks along d_x dimension

module attention_int #(
    parameter S_q = 4, // sequence length for Q
    parameter S_kv = 4, // sequence length for K and V
    parameter d_kq = 8, // embedding dimension of K and Q
    parameter d_v = 8,  // embedding dimension of V

    parameter k = 2, // MX block size
    parameter bit_width = 8,
    parameter out_width = 8,
    parameter scale_width = 8
)(
    input  logic                          i_clk,
    input  logic signed   [bit_width-1:0]   Q_i  [S_q][d_kq],
    input  logic signed   [bit_width-1:0]   Kt_i [d_kq][S_kv], // already transposed
    input  logic signed   [bit_width-1:0]   V_i  [S_kv][d_v],
    input  logic        [scale_width-1:0] S_Q_i  [S_q][d_kq/k],
    input  logic        [scale_width-1:0] S_Kt_i [d_kq/k][S_kv], // already transposed
    input  logic        [scale_width-1:0] S_V_i  [S_kv][d_v/k],

    output logic signed [out_width-1:0]   R_o  [S_q][d_v],
    output logic        [scale_width-1:0] S_R_o  [S_q][d_v/k]
);

    // Q * K^T
    logic signed   [out_width-1:0]   QKt [S_q][S_kv];
    logic        [scale_width-1:0] S_QKt [S_q][S_kv];

    matmul_int #(
        .x_rows(S_q),
        .vec_elem_count(d_kq),
        .y_cols(S_kv),
        .k(k),
        .bit_width(bit_width),
        .out_width(out_width),
        .scale_width(scale_width)
    ) u_matmul_QK (
        .i_clk(i_clk),
        .A_i(Q_i),
        .B_i(Kt_i),
        .S_A_i(S_Q_i),
        .S_B_i(S_Kt_i),
        .C_o(QKt),
        .S_C_o(S_QKt)
    );

    // Q * K^T / sqrt(d_kq)
    // Let's assume d_kq is a power of 4 for simplicity, so sqrt(d_kq) is a power of 2,
    // then we can just do adjust the S_QK scale by subtracting log_2(sqrt(d_kq)) = log_2(d_kq) / 2
    localparam scale_shift_bits = $clog2(d_kq) / 2;
    logic [scale_width-1:0] S_QKt_scaled  [S_q][S_kv];

    for (genvar i = 0; i < S_q; i++) begin : scale_row_loop
        for (genvar j = 0; j < S_kv; j++) begin : scale_col_loop
            assign S_QKt_scaled[i][j] = S_QKt[i][j] - scale_shift_bits;
        end
    end

    // Requantise QKt | S_QKt_scaled into FP format for softmax
    // TODO

    // softmax(Q * K^T / sqrt(d_kq))
    // TODO

    // Quantise softmax_res back to INT format for matmul
    // TODO

    // softmax(Q * K^T / sqrt(d_kq)) * V
    matmul_int #(
        .x_rows(S_q),
        .vec_elem_count(S_kv),
        .y_cols(d_v),
        .k(k),
        .bit_width(bit_width),
        .out_width(out_width),
        .scale_width(scale_width)
    ) u_matmul_SMV (
        .i_clk(i_clk),
        .A_i(...),
        .B_i(V_i),
        .S_A_i(...),
        .S_B_i(S_V_i),
        .C_o(R_o),
        .S_C_o(S_R_o)
    );
    
endmodule

`endif // __ATTENTION_INT_SV__