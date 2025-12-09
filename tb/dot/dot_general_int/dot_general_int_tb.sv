module dot_general_int_tb();

    // Generate clock and reset.
    logic clk;
    logic rst_n;

    initial begin
        clk = 0;
        forever
            #5 clk = ~clk;
    end

    initial begin
        rst_n = 0;
        #10
        rst_n = 1;
    end

    // Parameters and functions.
    localparam bit_width = 8;
    localparam k = 2;
    localparam C = 4;
    localparam block_count = C / k;
    localparam out_width = 2*bit_width + $clog2(k);


    // DUT
    logic signed [bit_width-1:0] i_X [C];
    logic signed [bit_width-1:0] i_Y [C];
    logic                [8-1:0] i_S [block_count];
    logic                [8-1:0] i_T [block_count];
    logic signed [out_width-1:0] o_dp;
    logic signed         [8-1:0] o_scale;

    dot_general_int #(
        .C(C),
        .k(k),
        .bit_width(bit_width),
        .out_width(out_width)
    ) u_dot_general_int (
        .i_clk(clk),
        .i_X(i_X),
        .i_Y(i_Y),
        .i_S(i_S),
        .i_T(i_T),
        .o_dp(o_dp),
        .o_scale(o_scale)
    );

    int scale_exp_bias = 127;
    int cur_scale;
    int cur_dp;

    // Reference
    real o_dp_ref;


    initial begin
        #10

        $display("Starting -----");
        $display("Width Exp:    %d", bit_width);
        $display("C:            %d", C);
        $display("K:            %d", k);
        $display("Block Count:  %d", block_count);

        for (int i = 0; i < (1<<3); i++) begin

            for (int j = 0; j < C; j++) begin
                i_X[j] = $random;
                i_Y[j] = $random;
            end

            // Random scales in range 128 +- 4, i.e. [2^-4, 2^4]
            for (int j = 0; j < block_count; j++) begin
                i_S[j] = ($random % 9) - 4 + scale_exp_bias + 1;
                i_T[j] = ($random % 9) - 4 + scale_exp_bias + 1;
            end

            o_dp_ref = 0.0;
            for (int j = 0; j < block_count; j++) begin
                cur_scale = i_S[j] + i_T[j] - scale_exp_bias;
                $display("Current scale: %d", cur_scale);
                $display("Current scale (as 2^x): %f", $pow(2.0, cur_scale - scale_exp_bias));

                cur_dp = 0;
                for (int m = 0; m < k; m++) begin
                    cur_dp += $signed(i_X[j*k + m]) * $signed(i_Y[j*k + m]);
                end

                $display("Current dp: %d", cur_dp);
                $display("Current dp (scaled): %f", $pow(2.0, cur_scale - scale_exp_bias) * cur_dp);

                o_dp_ref += $pow(2.0, cur_scale - scale_exp_bias) * cur_dp;
                $display("Accumulated ref dp: %f", o_dp_ref);
            end

            #10

            if (($signed(o_dp) * $pow(2.0, o_scale - scale_exp_bias)) != o_dp_ref) begin
                $display("Failed on: %d", i);
                $display("X in:");
                for (int n = 0; n < C; n++) begin
                    $display("X[%0d]: %d", n, i_X[n]);
                end
                $display("Y in: ");
                for (int n = 0; n < C; n++) begin
                    $display("Y[%0d]: %d", n, i_Y[n]);
                end
                $display("S in:");
                for (int n = 0; n < block_count; n++) begin
                    $display("S[%0d]: %d", n, i_S[n]);
                end
                $display("T in:");
                for (int n = 0; n < block_count; n++) begin
                    $display("T[%0d]: %d", n, i_T[n]);
                end
                $display("dp out: %d", o_dp);
                $display("scale out: %d", o_scale);
                $display("Ref dp out: %f", o_dp_ref);
                $display("FAILED");
                $finish();
            end
        end

        $display("PASSED");
        $finish();
    end






endmodule
