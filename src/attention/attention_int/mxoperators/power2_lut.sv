`timescale 1ns / 1ps

module power2_lut #(
  parameter int DATA_IN_0_PRECISION_0  = 7,
  parameter int DATA_IN_0_PRECISION_1  = 6,  // unused
  parameter int DATA_OUT_0_PRECISION_0 = 10,
  parameter int DATA_OUT_0_PRECISION_1 = 8   // unused
)(
  input  logic [DATA_IN_0_PRECISION_0-1:0]  data_in_0,
  output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0
);

  localparam int DEPTH = 1 << DATA_IN_0_PRECISION_0;

  logic [DATA_OUT_0_PRECISION_0-1:0] rom [0:DEPTH-1];

  // Placeholder contents (NOT hardcoded to specific widths)
  // Generates pseudo-random data to prevent logic trimming while testing resources
  function automatic [DATA_OUT_0_PRECISION_0-1:0] init_val(input int unsigned i);
    int unsigned x;
    begin
      x = i;
      x ^= (x << 13);
      x ^= (x >> 7);
      x ^= (x << 17);
      init_val = x[DATA_OUT_0_PRECISION_0-1:0];
    end
  endfunction

  integer k;
  initial begin
    for (k = 0; k < DEPTH; k++) rom[k] = init_val(k);
  end

  always_comb data_out_0 = rom[data_in_0];

endmodule
