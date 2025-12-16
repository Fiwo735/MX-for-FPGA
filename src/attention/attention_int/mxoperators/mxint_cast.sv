module mxint_cast #(
    parameter IN_MAN_WIDTH = 16,
    parameter IN_MAN_FRAC_WIDTH = 4,
    parameter IN_EXP_WIDTH = 8,
    parameter OUT_MAN_WIDTH = 4,
    parameter OUT_EXP_WIDTH = 8,
    parameter BLOCK_SIZE = 1,
    parameter ROUND_BITS = 4
)(
    input logic clk,
    input logic rst,
    input logic [IN_MAN_WIDTH-1:0] mdata_in [BLOCK_SIZE-1:0],
    input logic [IN_EXP_WIDTH-1:0] edata_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [OUT_MAN_WIDTH-1:0] mdata_out [BLOCK_SIZE-1:0],
    output logic [OUT_EXP_WIDTH-1:0] edata_out, 
    output logic data_out_valid,
    input logic data_out_ready
);
    assign data_in_ready = 1'b1;
    assign data_out_valid = data_in_valid;
    assign edata_out = edata_in;

    genvar i;
    generate
        for (i=0; i<BLOCK_SIZE; i++) begin
            assign mdata_out[i] = mdata_in[i];
        end
    endgenerate
endmodule
