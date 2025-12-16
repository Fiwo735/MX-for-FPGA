module unpacked_mx_split2_with_data #(
    parameter DEPTH = 16,
    parameter MAN_WIDTH = 4,
    parameter EXP_WIDTH = 8,
    parameter IN_SIZE = 1
)(
    input logic clk,
    input logic rst,
    input logic [MAN_WIDTH-1:0] mdata_in [IN_SIZE-1:0],
    input logic [EXP_WIDTH-1:0] edata_in, 
    input logic data_in_valid,
    output logic data_in_ready,
    
    output logic [MAN_WIDTH-1:0] fifo_mdata_out [IN_SIZE-1:0],
    output logic [EXP_WIDTH-1:0] fifo_edata_out, 
    output logic fifo_data_out_valid,
    input logic fifo_data_out_ready,
    
    output logic [MAN_WIDTH-1:0] straight_mdata_out [IN_SIZE-1:0],
    output logic [EXP_WIDTH-1:0] straight_edata_out, 
    output logic straight_data_out_valid,
    input logic straight_data_out_ready
);
    assign data_in_ready = 1'b1;
    assign fifo_data_out_valid = data_in_valid;
    assign fifo_edata_out = edata_in;
    assign straight_data_out_valid = data_in_valid;
    assign straight_edata_out = edata_in;
    
    genvar i;
    generate
        for (i=0; i<IN_SIZE; i++) begin
            assign fifo_mdata_out[i] = mdata_in[i];
            assign straight_mdata_out[i] = mdata_in[i];
        end
    endgenerate
endmodule
