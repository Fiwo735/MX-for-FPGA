module mxint_div #(
    parameter DATA_DIVIDEND_PRECISION_0 = 16,
    parameter DATA_DIVIDEND_PRECISION_1 = 8,
    parameter DATA_DIVISOR_PRECISION_0 = 16,
    parameter DATA_DIVISOR_PRECISION_1 = 8,
    parameter DATA_QUOTIENT_PRECISION_0 = 16,
    parameter DATA_QUOTIENT_PRECISION_1 = 8,
    parameter BLOCK_SIZE = 1
)(
    input logic clk,
    input logic rst,
    
    input logic [DATA_DIVIDEND_PRECISION_0-1:0] mdividend_data [BLOCK_SIZE-1:0],
    input logic [DATA_DIVIDEND_PRECISION_1-1:0] edividend_data,
    input logic dividend_data_valid,
    output logic dividend_data_ready,
    
    input logic [DATA_DIVISOR_PRECISION_0-1:0] mdivisor_data [BLOCK_SIZE-1:0],
    input logic [DATA_DIVISOR_PRECISION_1-1:0] edivisor_data,
    input logic divisor_data_valid,
    output logic divisor_data_ready,
    
    output logic [DATA_QUOTIENT_PRECISION_0-1:0] mquotient_data [BLOCK_SIZE-1:0],
    output logic [DATA_QUOTIENT_PRECISION_1-1:0] equotient_data,
    output logic quotient_data_valid,
    input logic quotient_data_ready
);
    assign dividend_data_ready = 1'b1;
    assign divisor_data_ready = 1'b1;
    assign quotient_data_valid = dividend_data_valid;
    assign equotient_data = edividend_data; // Pass through dividend exp
    
    genvar i;
    generate
        for (i=0; i<BLOCK_SIZE; i++) begin
            // Pass through dividend mantissa as quotient
            assign mquotient_data[i] = mdividend_data[i];
        end
    endgenerate
endmodule
