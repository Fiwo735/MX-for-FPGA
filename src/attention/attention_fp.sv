`ifndef __ATTENTION_FP_SV__
`define __ATTENTION_FP_SV__

// MX Operators Includes (Shared)
// Note: MX operators are compiled via run_synth.tcl.

module attention_fp #(
    parameter S_q = 4, // sequence length for Q
    parameter S_kv = 4, // sequence length for K and V
    parameter d_kq = 8, // embedding dimension of K and Q
    parameter d_v = 8,  // embedding dimension of V

    parameter k = 2, // MX block size
    parameter scale_width = 8,

    // M1: After Q*K^T (Softmax Input)
    parameter M1_MAN_WIDTH = 8,
    parameter M1_EXP_WIDTH = 0,

    // M2: After Softmax (MatMul2 Input)
    parameter M2_MAN_WIDTH = 8,
    parameter M2_EXP_WIDTH = 0, 

    // M3: Output
    parameter M3_MAN_WIDTH = 8,
    parameter M3_EXP_WIDTH = 0,

    parameter string M1_USE_DSP = "auto",
    parameter string M2_USE_DSP = "auto",
    parameter string SOFTMAX_USE_DSP = "auto",

    parameter string ACCUM_METHOD1 = "Kulisch",
    parameter string ACCUM_METHOD2 = "Kulisch",
    parameter string ACCUM_METHOD3 = "Kulisch",

    parameter accumulator_width_1 = 32,
    parameter accumulator_width_2 = 32
)(
    input  logic                          i_clk,
    // Input Data
    input  logic signed   [M1_MAN_WIDTH+M1_EXP_WIDTH:0]   Q_i  [S_q][d_kq],
    input  logic signed   [M1_MAN_WIDTH+M1_EXP_WIDTH:0]   Kt_i [d_kq][S_kv], 
    input  logic signed   [M2_MAN_WIDTH+M2_EXP_WIDTH:0]   V_i  [S_kv][d_v], // V matches second stage
    
    input  logic        [scale_width-1:0] S_Q_i  [S_q][d_kq/k],
    input  logic        [scale_width-1:0] S_Kt_i [d_kq/k][S_kv], 
    input  logic        [scale_width-1:0] S_V_i  [S_kv][d_v/k],

    output logic signed [M3_MAN_WIDTH+M3_EXP_WIDTH:0]   R_o  [S_q][d_v], // Result in M3 format
    output logic        [scale_width-1:0] S_R_o  [S_q][d_v/k]
);

    // Derived Widths
    localparam BIT_WIDTH_1 = M1_MAN_WIDTH + 1 + M1_EXP_WIDTH;
    
    localparam BW_1 = 1 + M1_EXP_WIDTH + M1_MAN_WIDTH;
    localparam BW_2 = 1 + M2_EXP_WIDTH + M2_MAN_WIDTH;
    localparam BW_3 = 1 + M3_EXP_WIDTH + M3_MAN_WIDTH;


    // Q * K^T
    // Output of MatMul is large accumulator.
    // We define an OUT_WIDTH for the matmul.
    localparam MM1_OUT_WIDTH = accumulator_width_1;
    
    logic signed   [MM1_OUT_WIDTH-1:0]   QKt [S_q][S_kv];
    logic        [scale_width-1:0] S_QKt [S_q][S_kv];

    matmul_fp #(
        .x_rows(S_q),
        .vec_elem_count(d_kq),
        .y_cols(S_kv),
        .k(k),
        .bit_width(BW_1),
        .exp_width(M1_EXP_WIDTH),
        .man_width(M1_MAN_WIDTH),
        .out_width(MM1_OUT_WIDTH),
        .scale_width(scale_width),
        .USE_DSP(M1_USE_DSP),
        .ACCUM_METHOD(ACCUM_METHOD1),
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
    localparam scale_shift_bits = $clog2(d_kq) / 2;
    logic [scale_width-1:0] S_QKt_scaled  [S_q][S_kv];

    for (genvar i = 0; i < S_q; i++) begin : scale_row_loop
        for (genvar j = 0; j < S_kv; j++) begin : scale_col_loop
            assign S_QKt_scaled[i][j] = S_QKt[i][j] - scale_shift_bits;
        end
    end

    // Softmax Logic
    
    logic [1:0] sm_cnt;
    always_ff @(posedge i_clk) sm_cnt <= sm_cnt + 1;
    
    logic i_rst;
    assign i_rst = 1'b0;

    // Intermediate Storage
    logic signed [BW_2-1:0] soft_res [S_q][S_kv];
    logic [scale_width-1:0] soft_scale [S_q][S_kv/k];

    for (genvar i = 0; i < S_q; i++) begin : sm_inst
        logic [BW_1-1:0] m_in [1];
        logic [scale_width-1:0] e_in;
        logic [BW_1-1:0] m_out [1]; 
        logic [scale_width-1:0] e_out;
        logic v_out;
        logic r_in; 

        // Cast/Truncate MatMul output to M1 format for Softmax
        assign m_in[0] = QKt[i][sm_cnt][BW_1-1:0];
        assign e_in = S_QKt_scaled[i][sm_cnt];

        // Reuse mxint_softmax
        mxint_softmax #(
            .DATA_IN_0_PRECISION_0(BW_1), // Treating as bits
            .DATA_IN_0_PRECISION_1(scale_width),
            .DATA_IN_0_DIM(S_kv),
            .DATA_IN_0_PARALLELISM(1),
            .DATA_OUT_0_PRECISION_0(BW_1), // Output same width
            .DATA_OUT_0_PRECISION_1(scale_width),
            .DATA_OUT_0_DIM(S_kv),
            .DATA_OUT_0_PARALLELISM(1),
            .USE_DSP(SOFTMAX_USE_DSP)
        ) u_curr_softmax (
            .rst(i_rst),
            .clk(i_clk),
            .mdata_in_0(m_in),
            .edata_in_0(e_in),
            .data_in_0_valid(1'b1),
            .data_in_0_ready(r_in),
            .mdata_out_0(m_out),
            .edata_out_0(e_out),
            .data_out_0_valid(v_out),
            .data_out_0_ready(1'b1)
        );

        // Capture and Cast to M2 (MatMul 2 Input)
        always_ff @(posedge i_clk) begin
             if (v_out) begin
                 // Simplified Cast M1 -> M2
                 soft_res[i][sm_cnt] <= m_out[0][BW_2-1:0]; 
                 soft_scale[i][sm_cnt[$clog2(S_kv)-1:$clog2(k)]] <= e_out; 
             end
        end
    end

    // Reshape scales for MatMul
    logic [scale_width-1:0] S_V_reshaped [S_kv/k][d_v];
    always_comb begin
        {>>{S_V_reshaped}} = {>>{S_V_i}};
    end

    // MatMul 2
    localparam MM2_OUT_WIDTH = accumulator_width_2;
    logic signed   [MM2_OUT_WIDTH-1:0]   Res_Raw [S_q][d_v];
    logic        [scale_width-1:0] S_Res_Raw [S_q][d_v];

    matmul_fp #(
        .x_rows(S_q),
        .vec_elem_count(S_kv),
        .y_cols(d_v),
        .k(k),
        .bit_width(BW_2),
        .exp_width(M2_EXP_WIDTH),
        .man_width(M2_MAN_WIDTH),
        .out_width(MM2_OUT_WIDTH),
        .scale_width(scale_width),
        .USE_DSP(M2_USE_DSP),
        .ACCUM_METHOD(ACCUM_METHOD3),
    ) u_matmul_SMV (
        .i_clk(i_clk),
        .A_i(soft_res),
        .B_i(V_i),
        .S_A_i(soft_scale),
        .S_B_i(S_V_reshaped),
        .C_o(Res_Raw),
        .S_C_o(S_Res_Raw)
    );
    
    // Final Cast to M3 (Output)
    for (genvar i = 0; i < S_q; i++) begin : out_map_row
        for (genvar j = 0; j < d_v; j++) begin : out_map_col
            assign R_o[i][j] = Res_Raw[i][j][BW_3-1:0]; // Simplified Cast
        end
    end

    // Output Scales (Subsampled)
    logic [scale_width-1:0] S_R_full [S_q][d_v];
    // Assign S_Res_Raw to S_R_full
     for (genvar i = 0; i < S_q; i++) begin : scale_full_assign_row
        for (genvar j = 0; j < d_v; j++) begin : scale_full_assign_col
            assign S_R_full[i][j] = S_Res_Raw[i][j];
        end
    end

    for (genvar i = 0; i < S_q; i++) begin : scale_out_reshape_row
        for (genvar j = 0; j < d_v/k; j++) begin : scale_out_reshape_col
            assign S_R_o[i][j] = S_R_full[i][j*k];
        end
    end

endmodule

`endif // __ATTENTION_FP_SV__
