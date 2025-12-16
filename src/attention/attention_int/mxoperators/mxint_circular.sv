module mxint_circular #(
    parameter DATA_PRECISION_0 = 16,
    parameter DATA_PRECISION_1 = 8,
    parameter IN_NUM = 1,
    parameter REPEAT = 8,
    parameter BUFFER_SIZE = 1
)(
    input logic clk,
    input logic rst,
    input logic [DATA_PRECISION_0-1:0] mdata_in [IN_NUM-1:0],
    input logic [DATA_PRECISION_1-1:0] edata_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [DATA_PRECISION_0-1:0] mdata_out [IN_NUM-1:0],
    output logic [DATA_PRECISION_1-1:0] edata_out,
    output logic data_out_valid,
    input logic data_out_ready
);
    assign data_in_ready = 1'b1;
    assign data_out_valid = data_in_valid;
    assign edata_out = edata_in;
    
    genvar i;
    generate
        for (i=0; i<IN_NUM; i++) begin
            assign mdata_out[i] = mdata_in[i];
        end
    endgenerate
endmodule
