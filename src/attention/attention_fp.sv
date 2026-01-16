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
    parameter string M3_USE_DSP = "auto",
    parameter string SOFTMAX_USE_DSP = "auto",

    parameter string ACCUM_METHOD1 = "KULISCH",
    parameter string ACCUM_METHOD2 = "KULISCH",
    parameter string ACCUM_METHOD3 = "KULISCH",

    parameter accumulator_width_1 = 32,
    parameter accumulator_width_2 = 32
)(
    input  logic                          i_clk,
    // Input Data
    input  logic signed   [M1_MAN_WIDTH+M1_EXP_WIDTH:0]   Q_i  [S_q][d_kq],
    input  logic signed   [M1_MAN_WIDTH+M1_EXP_WIDTH:0]   Kt_i [d_kq][S_kv], 
    input  logic signed   [M3_MAN_WIDTH+M3_EXP_WIDTH:0]   V_i  [S_kv][d_v], // V matches second stage
    
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
    
    logic signed   [BW_1-1:0]   QKt [S_q][S_kv];
    logic        [scale_width-1:0] S_QKt [S_q][S_kv];

    matmul_fp #(
        .x_rows(S_q),
        .vec_elem_count(d_kq),
        .y_cols(S_kv),
        .k(k),
        .bit_width(BW_1),
        .exp_width(M1_EXP_WIDTH),
        .man_width(M1_MAN_WIDTH),
        .out_width(BW_1),
        .scale_width(scale_width),
        .USE_DSP(M1_USE_DSP),
        .ACCUM_METHOD(ACCUM_METHOD1)
    ) u_matmul_QK (
        .i_clk(i_clk),
        .A_i(Q_i),
        .B_i(Kt_i),
        .S_A_i(S_Q_i),
        .S_B_i(S_Kt_i),
        .C_o(QKt),
        .S_C_o(S_QKt)
    );

    // Simple down cast of S_QKt to accomodate k blocks
    logic        [scale_width-1:0] S_QKt_blocked [S_q][S_kv/k];
    always_comb begin
        for (int i = 0; i < S_q; i++) begin
            for (int j = 0; j < S_kv/k; j++) begin
                S_QKt_blocked[i][j] = S_QKt[i][j*k];
            end
        end
    end

    // Q * K^T / sqrt(d_kq)
    localparam scale_shift_bits = $clog2(d_kq) / 2;
    logic [scale_width-1:0] S_QKt_scaled  [S_q][S_kv/k];

    for (genvar i = 0; i < S_q; i++) begin : scale_row_loop
        for (genvar j = 0; j < S_kv/k; j++) begin : scale_col_loop
            assign S_QKt_scaled[i][j] = S_QKt_blocked[i][j] - scale_shift_bits;
        end
    end

    logic i_rst;
    assign i_rst = 1'b0;

    // Intermediate Storage
    logic signed [BW_3-1:0] soft_res [S_q][S_kv];
    logic [scale_width-1:0] soft_scale [S_q][S_kv/k];

    logic    [BW_2-1:0]   QKt_2 [S_q][S_kv];
    // Cast (BW_2 < BW_1) or 0-pad (BW_2 > BW_1) QKt into QKt_2 
    always_comb begin
        for (int i = 0; i < S_q; i++) begin
            for (int j = 0; j < S_kv; j++) begin
                if (BW_2 < BW_1) begin
                    QKt_2[i][j] = QKt[i][j][BW_2-1:0]; // Truncate
                end else begin
                    QKt_2[i][j] = {{(BW_2-BW_1){1'b0}}, QKt[i][j]}; // Zero-pad
                end
            end
        end
    end
    

    for (genvar i = 0; i < S_q; i++) begin : sm_inst
        for (genvar j = 0; j < S_kv/k; j++) begin : sm_inner
            // QKt [S_q][S_kv];
            // k elements from QKt
            // 1 element from S_QKt_scaled


            logic v_out;
            logic r_in; 
            
            // Reuse mxint_softmax
            
            // Temporary variables for explicit slicing (Avoids Vivado unpacked array slicing bugs)
            logic [BW_2-1:0] sm_in_slice [k];
            logic [BW_3-1:0] sm_out_slice [k];
            
            // Manually copy input slice
            for (genvar el = 0; el < k; el++) begin : in_slice_assign
                assign sm_in_slice[el] = QKt_2[i][j*k + el];
            end
            
            mxint_softmax #(
                .DATA_IN_0_PRECISION_0(BW_2), // Treating as bits
                .DATA_IN_0_PRECISION_1(scale_width),
                .DATA_IN_0_DIM(BW_3),
                .DATA_IN_0_PARALLELISM(k),
                .DATA_OUT_0_PRECISION_0(BW_3),
                .DATA_OUT_0_PRECISION_1(scale_width),
                .DATA_OUT_0_DIM(BW_3),
                .DATA_OUT_0_PARALLELISM(k),
                .USE_DSP(SOFTMAX_USE_DSP),
                .ACCUM_METHOD(ACCUM_METHOD2)
            ) u_curr_softmax (
                .rst(i_rst),
                .i_clk(i_clk),
                .mdata_in_0(sm_in_slice), // Connected via temp variable
                .edata_in_0(S_QKt_scaled[i][j]),
                .data_in_0_valid(1'b1),
                .data_in_0_ready(r_in),
                .mdata_out_0(sm_out_slice), // Output to temp variable
                .edata_out_0(soft_scale[i][j]),
                .data_out_0_valid(v_out),
                .data_out_0_ready(1'b1)
            );

            // Manually copy output slice back
            for (genvar el = 0; el < k; el++) begin : out_slice_assign
                 assign soft_res[i][j*k + el] = sm_out_slice[el];
            end


            // // Capture and Cast to M2 (MatMul 2 Input)
            // always_ff @(posedge i_clk) begin
            //      if (v_out) begin
            //          // Simplified Cast
            //         //  soft_res[i][sm_cnt] <= m_out[0]; 
            //          soft_scale[i][sm_cnt[$clog2(S_kv)-1:$clog2(k)]] <= e_out; 
            //      end
            // end
        end
    end

    // Reshape scales for MatMul
    logic [scale_width-1:0] S_V_reshaped [S_kv/k][d_v];
    always_comb begin
        {>>{S_V_reshaped}} = {>>{S_V_i}};
    end

    // MatMul 2
    localparam MM2_OUT_WIDTH = accumulator_width_2;
    // logic signed   [MM2_OUT_WIDTH-1:0]   Res_Raw [S_q][d_v];
    logic signed   [BW_3-1:0]   Res_Raw [S_q][d_v];
    logic        [scale_width-1:0] S_Res_Raw [S_q][d_v];

    matmul_fp #(
        .x_rows(S_q),
        .vec_elem_count(S_kv),
        .y_cols(d_v),
        .k(k),
        .bit_width(BW_3),
        .exp_width(M3_EXP_WIDTH),
        .man_width(M3_MAN_WIDTH),
        .out_width(BW_3),
        .scale_width(scale_width),
        .USE_DSP(M3_USE_DSP),
        .ACCUM_METHOD(ACCUM_METHOD3)
    ) u_matmul_SMV (
        .i_clk(i_clk),
        .A_i(soft_res),
        .B_i(V_i),
        .S_A_i(soft_scale),
        .S_B_i(S_V_reshaped),
        .C_o(Res_Raw),
        .S_C_o(S_Res_Raw)
    );
    
    // //////Final Cast to M3 (Output)
    // Final assign
    for (genvar i = 0; i < S_q; i++) begin : out_map_row
        for (genvar j = 0; j < d_v; j++) begin : out_map_col
            // assign R_o[i][j] = Res_Raw[i][j][BW_3-1:0]; // Simplified Cast
            assign R_o[i][j] = Res_Raw[i][j];
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
