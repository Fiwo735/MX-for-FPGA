module mxint_accumulator #(
    parameter DATA_IN_0_PRECISION_0 = 4,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter BLOCK_SIZE = 1,
    parameter IN_DEPTH = 8,
    parameter UNDERFLOW_BITS = 4
)(
    input logic clk,
    input logic rst,
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [BLOCK_SIZE-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,
    
    output logic [DATA_IN_0_PRECISION_0 + $clog2(IN_DEPTH) + UNDERFLOW_BITS - 1:0] mdata_out_0 [BLOCK_SIZE-1:0],
    output logic [DATA_IN_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
    assign data_in_0_ready = 1'b1;
    assign data_out_0_valid = data_in_0_valid;
    assign edata_out_0 = edata_in_0;
    
    genvar i;
    generate
        for (i=0; i<BLOCK_SIZE; i++) begin
            assign mdata_out_0[i] = mdata_in_0[i];
        end
    endgenerate
endmodule
