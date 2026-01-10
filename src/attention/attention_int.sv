`ifndef __ATTENTION_INT_SV__
`define __ATTENTION_INT_SV__


`include "../matmul/matmul_int/matmul_int.sv"

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
    // Find max scale per row needed for MX format alignment
    logic [scale_width-1:0] row_max_exp [S_q];
    logic signed [out_width-1:0] QKt_aligned [S_q][S_kv];

    always_comb begin
        for (int i = 0; i < S_q; i++) begin
            row_max_exp[i] = S_QKt_scaled[i][0];
            for (int j = 1; j < S_kv; j++) begin
                if (S_QKt_scaled[i][j] > row_max_exp[i]) begin
                    row_max_exp[i] = S_QKt_scaled[i][j];
                end
            end
        end
    end

    always_comb begin
        for (int i = 0; i < S_q; i++) begin
            for (int j = 0; j < S_kv; j++) begin
                // Shift right by difference in exponents (align to max exponent)
                QKt_aligned[i][j] = QKt[i][j] >>> (row_max_exp[i] - S_QKt_scaled[i][j]);
            end
        end
    end

    // softmax(Q * K^T / sqrt(d_kq))
    logic signed [bit_width-1:0] softmax_m [S_q][S_kv];
    logic [scale_width-1:0] softmax_e [S_q];
    
    genvar gi;
    generate
        for (gi = 0; gi < S_q; gi++) begin : gen_softmax
            
            logic [out_width-1:0] row_in [S_kv];
            logic [out_width-1:0] row_out [S_kv];
            
            for (genvar gj = 0; gj < S_kv; gj++) begin : gen_softmax_io
                 assign row_in[gj] = QKt_aligned[gi][gj];
                 assign softmax_m[gi][gj] = row_out[gj];
            end

            mxint_softmax #(
                .DATA_IN_0_PRECISION_0(out_width),
                .DATA_IN_0_PRECISION_1(scale_width),
                .DATA_IN_0_DIM(S_kv),
                .DATA_IN_0_PARALLELISM(S_kv), // Parallel input of full vector
                .DATA_R_WIDTH(2),
                .DATA_OUT_0_PRECISION_0(out_width),
                .DATA_OUT_0_PRECISION_1(scale_width)
            ) u_softmax_row (
                .rst(1'b0), // No reset available
                .clk(i_clk),
                .mdata_in_0(row_in),
                .edata_in_0(row_max_exp[gi]),
                .mdata_out_0(row_out),
                .edata_out_0(softmax_e[gi]),
                .data_in_0_valid(1'b1),
                .data_in_0_ready(),
                .data_out_0_valid(),
                .data_out_0_ready(1'b1)
            );
        end
    endgenerate

    // Quantise softmax_res back to INT format for matmul
    // Broadcast the shared exponent from softmax to the scale input of matmul
    logic [scale_width-1:0] S_softmax [S_q][S_kv/k];
    
    generate 
        for (gi = 0; gi < S_q; gi++) begin : gen_scale_broadcast
            for (genvar gj = 0; gj < S_kv/k; gj++) begin : gen_scale_cols
                assign S_softmax[gi][gj] = softmax_e[gi];
            end
        end
    endgenerate

    // Adapt S_V_i [S_kv][d_v/k] to matmul expected S_B_i [S_kv/k][d_v]
    // Mismatch: S_V_i is blocked on cols, matmul wants blocked on rows.
    logic [scale_width-1:0] S_V_adapted [S_kv/k][d_v];
    
    always_comb begin
        for (int i = 0; i < S_kv/k; i++) begin 
             for (int j = 0; j < d_v; j++) begin 
                  // Find max scale in the block of rows [i*k, i*k + k - 1]
                  logic [scale_width-1:0] max_s;
                  max_s = S_V_i[i*k][j/k];
                  for (int r = 1; r < k; r++) begin
                      if (S_V_i[i*k + r][j/k] > max_s) begin
                          max_s = S_V_i[i*k + r][j/k];
                      end
                  end
                  S_V_adapted[i][j] = max_s;
             end
        end
    end

    // softmax(Q * K^T / sqrt(d_kq)) * V
    logic [scale_width-1:0] S_R_full [S_q][d_v];

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
        .A_i(softmax_m),
        .B_i(V_i),
        .S_A_i(S_softmax),
        .S_B_i(S_V_adapted),
        .C_o(R_o),
        .S_C_o(S_R_full)
    );

    // Reduce S_R_full [S_q][d_v] to S_R_o [S_q][d_v/k]
    // Since we duplicated input scales for B, output scales should be identical within blocks of k.
    // So we can just pick the first one.
    generate
        for (gi = 0; gi < S_q; gi++) begin : gen_sr_rows
             for (genvar gj = 0; gj < d_v/k; gj++) begin : gen_sr_cols
                  assign S_R_o[gi][gj] = S_R_full[gi][gj*k];
             end
        end
    endgenerate
    
endmodule

`endif // __ATTENTION_INT_SV__