`timescale 1ns / 1ps

module fixed_round #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 4,
    parameter OUT_FRAC_WIDTH = 2
) (
    input  logic [IN_WIDTH-1:0] data_in,
    output logic [OUT_WIDTH-1:0] data_out
);

  localparam INT_WIDTH_IN = IN_WIDTH - IN_FRAC_WIDTH;
  localparam INT_WIDTH_OUT = OUT_WIDTH - OUT_FRAC_WIDTH;

  // Signed calculation
  logic signed [IN_WIDTH-1:0] signed_in;
  assign signed_in = data_in;

  logic signed [IN_WIDTH:0] rounded; // +1 bit for overflow
  logic signed [OUT_WIDTH-1:0] final_out;

  // Rounding logic: Add 0.5 (LSB of target precision) and truncate?
  // Or usually: if we drop bits, we check the MSB of dropped bits.
  
  localparam DIFF_FRAC = IN_FRAC_WIDTH - OUT_FRAC_WIDTH;
  
  generate
    if (DIFF_FRAC > 0) begin : gen_round
      // We are reducing precision (dropping LSBs)
      logic [IN_WIDTH-1:0] half_lsb;
      assign half_lsb = (1 << (DIFF_FRAC - 1));
      
      // Perform rounding: Add half LSB
      assign rounded = signed_in + half_lsb;
      
      // Truncate and Clamp
      // Select the bits corresponding to the new format
      // integer part + fractional part
      // The bit position of the new LSB in the 'rounded' result is DIFF_FRAC
      
      logic signed [IN_WIDTH - DIFF_FRAC:0] temp_rounded;
      assign temp_rounded = rounded[IN_WIDTH:DIFF_FRAC];
      
      // Now clamp to output range
      signed_clamp #(
        .IN_WIDTH(IN_WIDTH - DIFF_FRAC + 1), // width of temp_rounded
        .OUT_WIDTH(OUT_WIDTH),
        .SYMMETRIC(0)
      ) clamp_inst (
        .in_data(temp_rounded),
        .out_data(final_out)
      );
      
    end else begin : gen_pad
       // We are increasing or keeping precision
       // Just pad with zeros or clamp?
       // Usually just clamp if int part changes, shift if frac changes.
       // Assuming simplistic padding/clamping for now.
       assign final_out = signed_in <<< (-DIFF_FRAC);
    end
  endgenerate

  assign data_out = final_out;

endmodule
