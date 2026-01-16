`timescale 1ns / 1ps
/*
  Currently, we dont' want to support parallelism
  Cause in attention, it's actually not in parallel
*/
module mxint_softmax #(
    /* verilator lint_off UNUSEDPARAM */

    parameter DATA_IN_0_PRECISION_0 = 4,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter DATA_IN_0_DIM = 8,  // input vector size
    parameter DATA_IN_0_PARALLELISM = 1,  // batch size
    parameter DATA_R_WIDTH = 2,

    parameter IN_0_DEPTH = DATA_IN_0_DIM,
    parameter DATA_OUT_0_PRECISION_0 = 4,
    parameter DATA_OUT_0_PRECISION_1 = 8,
    parameter DATA_OUT_0_DIM = DATA_IN_0_DIM,
    parameter DATA_OUT_0_PARALLELISM = DATA_IN_0_PARALLELISM,
    parameter EXP_SUM_UNDERFLOW_BITS = 0,
    parameter DIVISION_UNDERFLOW_BITS = 0,
    parameter string USE_DSP = "auto",
    parameter string ACCUM_METHOD = "KULISCH"
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input i_clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[DATA_IN_0_PARALLELISM-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0[DATA_OUT_0_PARALLELISM-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  // softmax over a vector
  // each vector might be split into block of elements
  // Can handle multiple batches at once
  // each iteration recieves a batch of blocks

  // The current version only support precision of taylor_exp output to be the same with data_out_r
  localparam DATA_EXP_0_PRECISION_0 = DATA_OUT_0_PRECISION_0;
  localparam DATA_EXP_0_FRAC_WIDTH = DATA_EXP_0_PRECISION_0 - 2;
  localparam DATA_EXP_0_PRECISION_1 = DATA_OUT_0_PRECISION_1;

  localparam ACC_WIDTH = DATA_EXP_0_PRECISION_0;
  localparam ACC_FRAC_WIDTH = DATA_EXP_0_FRAC_WIDTH + EXP_SUM_UNDERFLOW_BITS;

  localparam DATA_DIVIDEND_PRECISION_0 = DATA_EXP_0_PRECISION_0 + EXP_SUM_UNDERFLOW_BITS + DIVISION_UNDERFLOW_BITS;
  localparam DATA_DIVIDEND_PRECISION_1 = DATA_EXP_0_PRECISION_1;
  localparam DATA_DIVISOR_PRECISION_0 = ACC_WIDTH;
  localparam DATA_DIVISOR_PRECISION_1 = DATA_EXP_0_PRECISION_1;
  localparam DATA_QUOTIENT_PRECISION_0 = DATA_OUT_0_PRECISION_0;
  localparam DATA_QUOTIENT_FRAC_WIDTH = DIVISION_UNDERFLOW_BITS;
  localparam DATA_QUOTIENT_PRECISION_1 = DATA_EXP_0_PRECISION_1 + 1;


  localparam BLOCK_SIZE = DATA_IN_0_PARALLELISM;
  initial begin
    // assert (BLOCK_SIZE == 1)
    // else $fatal(1, "Currently only BLOCK_SIZE of 1 is supported.");
  end

  // Add missing signals for mxint_exp interface
  logic [DATA_IN_0_PRECISION_0-1:0] mdata_exp[BLOCK_SIZE - 1:0];
  logic [DATA_IN_0_PRECISION_1-1:0] edata_exp[BLOCK_SIZE - 1:0];
  logic data_exp_valid, data_exp_ready;

  // New intermediate signals for Cast Output
  logic [DATA_EXP_0_PRECISION_0-1:0] cast_mdata[BLOCK_SIZE - 1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] cast_edata[BLOCK_SIZE - 1:0];
  logic cast_data_valid, cast_data_ready;

  // Split2 and FF signals for exp path
  logic [DATA_EXP_0_PRECISION_0-1:0] ff_exp_mdata_out[DATA_IN_0_PARALLELISM-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] ff_exp_edata_out;
  logic ff_exp_data_valid, ff_exp_data_ready;

  // Straight path signals
  // Output from Split2 (Descending/Unsigned)
  logic [DATA_EXP_0_PRECISION_0-1:0] straight_exp_mdata_from_split[DATA_IN_0_PARALLELISM-1:0];
  // Adapted for Accumulator (Ascending/Signed)
  logic signed [DATA_EXP_0_PRECISION_0-1:0] straight_exp_mdata_out[DATA_IN_0_PARALLELISM];
  logic [DATA_EXP_0_PRECISION_1-1:0] straight_exp_edata_out;
  logic straight_exp_data_out_valid, straight_exp_data_out_ready;

  // Adapt Split2 output to Accumulator input
  always_comb begin
      for (int i = 0; i < DATA_IN_0_PARALLELISM; i++) begin
          straight_exp_mdata_out[i] = $signed(straight_exp_mdata_from_split[i]);
      end
  end

  // Accumulator signals
  logic [ACC_WIDTH-1:0] acc_mdata_out[BLOCK_SIZE-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] acc_edata_out;
  logic acc_data_out_valid, acc_data_out_ready;

  // Circular buffer signals
  logic [ACC_WIDTH-1:0] circ_mdata_out[DATA_OUT_0_PARALLELISM-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] circ_edata_out;
  logic circ_data_out_valid, circ_data_out_ready;

  logic [DATA_DIVIDEND_PRECISION_0 - 1:0] mdata_dividend [BLOCK_SIZE - 1:0];
  logic [DATA_DIVIDEND_PRECISION_1 - 1:0] edata_dividend;
  // Division signals
  logic [DATA_QUOTIENT_PRECISION_0 - 1:0] mquotient_data[BLOCK_SIZE - 1:0];
  logic [DATA_QUOTIENT_PRECISION_1 - 1:0] equotient_data;
  logic quotient_data_valid, quotient_data_ready;

  // Updated mxint_exp instantiation with all parameters and proper signal connections
  mxint_exp #(
      .DATA_IN_MAN_WIDTH(DATA_IN_0_PRECISION_0),
      .DATA_IN_EXP_WIDTH(DATA_IN_0_PRECISION_1),
      .BLOCK_SIZE(BLOCK_SIZE),
      .DATA_R_WIDTH(DATA_R_WIDTH),
      .DATA_OUT_MAN_WIDTH(DATA_IN_0_PRECISION_0), // Output Input Width (High Precision)
      .DATA_OUT_EXP_WIDTH(DATA_IN_0_PRECISION_1), 
      .USE_DSP(USE_DSP)
  ) mxint_exp_inst (
      .rst(rst),
      .clk(i_clk),
      // Input interface
      .mdata_in_0(mdata_in_0),
      .edata_in_0(edata_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),
      // Output interface
      .mdata_out_0(mdata_exp), // High Precision Out
      .edata_out_0(edata_exp),
      .data_out_0_valid(data_exp_valid),
      .data_out_0_ready(data_exp_ready)
  );

  // New Cast Instantiation
  mxint_cast #(
      .IN_MAN_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_MAN_FRAC_WIDTH(DATA_IN_0_PRECISION_0 - 2), 
      .IN_EXP_WIDTH(DATA_IN_0_PRECISION_1),
      .OUT_MAN_WIDTH(DATA_EXP_0_PRECISION_0), // Target Output Width
      .OUT_EXP_WIDTH(DATA_EXP_0_PRECISION_1),
      .BLOCK_SIZE(BLOCK_SIZE),
      .ROUND_BITS(4)
  ) cast_exp_inst (
      .clk(i_clk),
      .rst(rst),
      .mdata_in(mdata_exp),
      .edata_in(edata_exp[0]), // Assuming uniform exponent or taking 0
      .data_in_valid(data_exp_valid),
      .data_in_ready(data_exp_ready),
      .mdata_out(cast_mdata),
      .edata_out(cast_edata[0]),
      .data_out_valid(cast_data_valid),
      .data_out_ready(cast_data_ready)
  );

  unpacked_mx_split2_with_data #(
      .DEPTH(DATA_IN_0_DIM * 2),
      .MAN_WIDTH(DATA_EXP_0_PRECISION_0), // Uses Low Precision
      .EXP_WIDTH(DATA_EXP_0_PRECISION_1),
      .IN_SIZE(DATA_IN_0_PARALLELISM)
  ) split2_mxint_exp_inst (
      .clk(i_clk),
      .rst(rst),
      // Input from CAST (Low Precision)
      .mdata_in(cast_mdata),
      .edata_in(cast_edata[0]),
      .data_in_valid(cast_data_valid),
      .data_in_ready(cast_data_ready),
      // FIFO output path
      .fifo_mdata_out(ff_exp_mdata_out),
      .fifo_edata_out(ff_exp_edata_out),  // Not used
      .fifo_data_out_valid(ff_exp_data_valid),
      .fifo_data_out_ready(ff_exp_data_ready),
      // Straight output path
      .straight_mdata_out(straight_exp_mdata_from_split),
      .straight_edata_out(straight_exp_edata_out),
      .straight_data_out_valid(straight_exp_data_out_valid),
      .straight_data_out_ready(straight_exp_data_out_ready)
  );

  // Intermediate scalar sum for broadcasting
  // Intermediate scalar sum for broadcasting to all lanes
  // "tmp" holds the single result from the adder tree before it is distributed
  logic [ACC_WIDTH-1:0] acc_mdata_tmp;

  generate
        if (ACCUM_METHOD == "KULISCH") begin : gen_kulisch_accum
            // [SKIPPED] Kulisch logic requires more careful handling or revert to accumulator
            // But implementing broadcast fix here too for consistency with request
            localparam KULISCH_BITWIDTH = 1 << (DATA_EXP_0_PRECISION_0 + 1);
            logic signed [KULISCH_BITWIDTH-1:0] KULISCH_INPUT [DATA_IN_0_PARALLELISM];

            for (genvar i=0; i<DATA_IN_0_PARALLELISM; i++) begin
                for (genvar j=0; j<KULISCH_BITWIDTH; j++) begin
                    assign KULISCH_INPUT[i][j] = straight_exp_mdata_out[i][j % DATA_EXP_0_PRECISION_0];
                end
            end

            vec_sum_int #(
                .bit_width(KULISCH_BITWIDTH),
                .length(DATA_IN_0_PARALLELISM),
                .sum_width(ACC_WIDTH) // Enforce output width
            ) u_tree_add (
                .i_clk(i_clk),            // FIX: Connected clock 
                .i_vec(KULISCH_INPUT),
                .o_sum(acc_mdata_tmp)
            );
        end else begin : gen_other_accum
            // Use other accumulation methods
            if (ACCUM_METHOD == "KAHAN") begin : gen_kahan_accum
                kahan_adder_tree #(
                    .EXP_WIDTH_I(0),
                    .MANT_WIDTH_I(DATA_EXP_0_PRECISION_0 - 1),
                    .ELEMS_COUNT(DATA_IN_0_PARALLELISM),
                    .SUM_WIDTH_O(ACC_WIDTH)
                ) u_kahan_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(straight_exp_mdata_out),
                    .o_sum(acc_mdata_tmp)
                );
            end else if (ACCUM_METHOD == "TWOSUM") begin : gen_twosum_accum
                twosum_adder_tree #(
                    .EXP_WIDTH_I(0),
                    .MANT_WIDTH_I(DATA_EXP_0_PRECISION_0 - 1),
                    .ELEMS_COUNT(DATA_IN_0_PARALLELISM),
                    .SUM_WIDTH_O(ACC_WIDTH)
                ) u_twosum_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(straight_exp_mdata_out),
                    .o_sum(acc_mdata_tmp)
                );
            end else if (ACCUM_METHOD == "FASTTWOSUM") begin : gen_fasttwosum_accum
                fasttwosum_adder_tree #(
                    .EXP_WIDTH_I(0),
                    .MANT_WIDTH_I(DATA_EXP_0_PRECISION_0 - 1),
                    .ELEMS_COUNT(DATA_IN_0_PARALLELISM),
                    .SUM_WIDTH_O(ACC_WIDTH)
                ) u_fasttwosum_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(straight_exp_mdata_out),
                    .o_sum(acc_mdata_tmp)
                );
            end else if (ACCUM_METHOD == "NEUMAIER") begin : gen_neumaier_accum
                neumaier_adder_tree #(
                    .EXP_WIDTH_I(0),
                    .MANT_WIDTH_I(DATA_EXP_0_PRECISION_0 - 1),
                    .ELEMS_COUNT(DATA_IN_0_PARALLELISM),
                    .SUM_WIDTH_O(ACC_WIDTH)
                ) u_neumaier_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(straight_exp_mdata_out),
                    .o_sum(acc_mdata_tmp)
                );
            end else if (ACCUM_METHOD == "KLEIN") begin : gen_klein_accum
                klein_adder_tree #(
                    .EXP_WIDTH_I(0),
                    .MANT_WIDTH_I(DATA_EXP_0_PRECISION_0 - 1),
                    .ELEMS_COUNT(DATA_IN_0_PARALLELISM),
                    .SUM_WIDTH_O(ACC_WIDTH)
                ) u_klein_tree (
                    .clk_i(i_clk),
                    .rst_ni(1'b1), // No reset
                    .i_vec(straight_exp_mdata_out),
                    .o_sum(acc_mdata_tmp)
                );
            end else begin : gen_error_accum
                $error("Unsupported ACCUM_METHOD");
            end
        end
    endgenerate

    // Broadcast scalar sum to all array elements
    generate
        for(genvar k=0; k<DATA_OUT_0_PARALLELISM; k++) begin
            assign acc_mdata_out[k] = acc_mdata_tmp;
        end
    endgenerate
    
    // LATENCY COMPENSATION
    // 1. Acc latency from vec_sum_int = $clog2(DATA_IN_0_PARALLELISM)
    // 2. We use delay line to match valid/edata with the Acc latency
    
    localparam ACCUM_LATENCY = $clog2(DATA_IN_0_PARALLELISM);
    
    logic [ACCUM_LATENCY:0] valid_delay_line; // Width is depth
    logic [DATA_EXP_0_PRECISION_1-1:0] edata_delay_line [ACCUM_LATENCY+1]; // Need Array for data
    
    always_ff @(posedge i_clk) begin
        if (rst) begin
             valid_delay_line <= '0;
        end else begin
             // Shift register for valid
             valid_delay_line[0] <= straight_exp_data_out_valid;
             if (ACCUM_LATENCY > 0) begin
                 for(int l=1; l<=ACCUM_LATENCY; l++) begin
                     valid_delay_line[l] <= valid_delay_line[l-1];
                 end
             end
        end
        
        // Shift register for data
        edata_delay_line[0] <= straight_exp_edata_out;
        if (ACCUM_LATENCY > 0) begin
             for(int l=1; l<=ACCUM_LATENCY; l++) begin
                 edata_delay_line[l] <= edata_delay_line[l-1];
             end
        end
    end

    // Use delayed version
    assign acc_edata_out = (ACCUM_LATENCY > 0) ? edata_delay_line[ACCUM_LATENCY] : straight_exp_edata_out;
    assign acc_data_out_valid = (ACCUM_LATENCY > 0) ? valid_delay_line[ACCUM_LATENCY] : straight_exp_data_out_valid; 
    
    assign acc_data_out_ready = 1'b1; // Always ready? Need to check flow control


//   mxint_accumulator #(
//       .DATA_IN_0_PRECISION_0(DATA_EXP_0_PRECISION_0),
//       .DATA_IN_0_PRECISION_1(DATA_EXP_0_PRECISION_1),
//       .BLOCK_SIZE(DATA_OUT_0_PARALLELISM),
//       .IN_DEPTH(IN_0_DEPTH),
//       .UNDERFLOW_BITS(EXP_SUM_UNDERFLOW_BITS),
//       .DATA_OUT_0_PRECISION_0(DATA_EXP_0_PRECISION_0)
//   ) mxint_accumulator_inst (
//       .clk(i_clk),
//       .rst(rst),
//       .mdata_in_0(straight_exp_mdata_out),     // From split2 straight output
//       .edata_in_0(straight_exp_edata_out),     // From split2 straight output
//       .data_in_0_valid(straight_exp_data_out_valid),
//       .data_in_0_ready(straight_exp_data_out_ready),
//       .mdata_out_0(acc_mdata_out),
//       .edata_out_0(acc_edata_out),
//       .data_out_0_valid(acc_data_out_valid),
//       .data_out_0_ready(acc_data_out_ready)
//   );
  // Replace existing signals
  // Replace input_buffer with mxint_circular
  mxint_circular #(
      .DATA_PRECISION_0(ACC_WIDTH),
      .DATA_PRECISION_1(DATA_EXP_0_PRECISION_1),
      .IN_NUM(DATA_OUT_0_PARALLELISM),
      .REPEAT(IN_0_DEPTH),
      .BUFFER_SIZE(1)
  ) acc_circular (
      .clk(i_clk),
      .rst(rst),
      .mdata_in(acc_mdata_out),
      .edata_in(acc_edata_out),
      .data_in_valid(acc_data_out_valid),
      .data_in_ready(acc_data_out_ready),
      .mdata_out(circ_mdata_out),
      .edata_out(circ_edata_out),
      .data_out_valid(circ_data_out_valid),
      .data_out_ready(circ_data_out_ready)
  );

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : dividend
    assign mdata_dividend[i] = ff_exp_mdata_out[i] << EXP_SUM_UNDERFLOW_BITS + DIVISION_UNDERFLOW_BITS;
  end
    assign edata_dividend = ff_exp_edata_out;
  // Add after mxint_circular instance
  mxint_div #(
      .DATA_DIVIDEND_PRECISION_0(DATA_DIVIDEND_PRECISION_0),
      .DATA_DIVIDEND_PRECISION_1(DATA_DIVIDEND_PRECISION_1),
      .DATA_DIVISOR_PRECISION_0(DATA_DIVISOR_PRECISION_0),
      .DATA_DIVISOR_PRECISION_1(DATA_DIVISOR_PRECISION_1),
      .DATA_QUOTIENT_PRECISION_0(DATA_QUOTIENT_PRECISION_0),
      .DATA_QUOTIENT_PRECISION_1(DATA_QUOTIENT_PRECISION_1),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM)
  ) div_inst (
      .clk(i_clk),
      .rst(rst),
      // Connect dividend (ff_exp_data)
      .mdividend_data(mdata_dividend),
      .edividend_data(edata_dividend),
      .dividend_data_valid(ff_exp_data_valid),
      .dividend_data_ready(ff_exp_data_ready),
      // Connect divisor (circ_data)
      .mdivisor_data(circ_mdata_out),
      .edivisor_data(circ_edata_out),
      .divisor_data_valid(circ_data_out_valid),
      .divisor_data_ready(circ_data_out_ready),
      // Connect quotient output directly to Module Output
      .mquotient_data(mdata_out_0), 
      .equotient_data(edata_out_0), 
      .quotient_data_valid(data_out_0_valid),
      .quotient_data_ready(data_out_0_ready)
  );


endmodule
