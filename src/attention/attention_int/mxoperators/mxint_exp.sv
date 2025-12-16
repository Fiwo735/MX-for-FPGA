module mxint_exp #(
    parameter DATA_IN_MAN_WIDTH = 4,
    parameter DATA_IN_EXP_WIDTH = 8,
    parameter BLOCK_SIZE = 1,
    parameter DATA_R_WIDTH = 2,
    parameter DATA_OUT_MAN_WIDTH = 4,
    parameter DATA_OUT_EXP_WIDTH = 8
)(
    input logic rst,
    input logic clk,
    input logic [DATA_IN_MAN_WIDTH-1:0] mdata_in_0 [BLOCK_SIZE-1:0],
    input logic [DATA_IN_EXP_WIDTH-1:0] edata_in_0, 
    input logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic [DATA_OUT_MAN_WIDTH-1:0] mdata_out_0 [BLOCK_SIZE-1:0],
    output logic [DATA_OUT_EXP_WIDTH-1:0] edata_out_0 [BLOCK_SIZE-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
    // Pass-through with resizing
    assign data_in_0_ready = 1'b1;
    assign data_out_0_valid = data_in_0_valid;
    
    genvar i;
    generate
        for (i=0; i<BLOCK_SIZE; i++) begin
            assign mdata_out_0[i] = mdata_in_0[i]; // Resize happens automatically
            assign edata_out_0[i] = edata_in_0;    // Broadcast scalar exp
        end
    endgenerate

endmodule
