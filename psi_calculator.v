
module psi_calculator (
    input clk,
    input rst,
    input [7:0] ar,    // Atmospheric light R component (0-255)
    input [7:0] ag,    // Atmospheric light G component (0-255)
    input [7:0] ab,    // Atmospheric light B component (0-255)
    input valid_in,
    output reg [31:0] psi_scaled, // PSI * 1,000,000 (divide by 1M to get actual PSI)
    output reg valid_out
);

    // Fixed coefficients (scaled by 1,000,000)
    parameter INTERCEPT = 32'd771580;
    parameter COEFF_AR = 32'd18641;
    parameter COEFF_AG = -32'd29403;  // Negative coefficient
    parameter COEFF_AB = 32'd12765;

    // Internal signals
    reg [31:0] ar_term, ag_term, ab_term;
    reg [31:0] sum;
    reg valid_d1, valid_d2;

    always @(posedge clk) begin
        if (rst) begin
            psi_scaled <= 32'd0;
            valid_out <= 1'b0;
            valid_d1 <= 1'b0;
            valid_d2 <= 1'b0;
        end else begin
            // Pipeline stage 1: Calculate individual terms
            ar_term <= ar * COEFF_AR;
            ag_term <= ag * COEFF_AG;  // This will be negative
            ab_term <= ab * COEFF_AB;
            valid_d1 <= valid_in;

            // Pipeline stage 2: Sum all terms
            sum <= INTERCEPT + ar_term + ag_term + ab_term;
            valid_d2 <= valid_d1;

            // Pipeline stage 3: Output result
            psi_scaled <= sum;
            valid_out <= valid_d2;
        end
    end

endmodule

// Alternative lookup-table based implementation for minimal area
module psi_calculator_lut (
    input [7:0] ar,
    input [7:0] ag,
    input [7:0] ab,
    output reg [31:0] psi_scaled
);

    // Convert RGB to range (0=low <200, 1=mid 200-239, 2=high >=240)
    wire [1:0] ar_range = (ar < 200) ? 2'd0 : (ar < 240) ? 2'd1 : 2'd2;
    wire [1:0] ag_range = (ag < 200) ? 2'd0 : (ag < 240) ? 2'd1 : 2'd2;
    wire [1:0] ab_range = (ab < 200) ? 2'd0 : (ab < 240) ? 2'd1 : 2'd2;

    wire [5:0] lut_addr = {ar_range, ag_range, ab_range};

    always @(*) begin
        case (lut_addr)
            6'b000000: psi_scaled = 32'd1510000; // Low RGB
            6'b000001: psi_scaled = 32'd1400000; // Low RG, Mid B
            6'b000011: psi_scaled = 32'd1240000; // Low R, Mid GB
            6'b010000: psi_scaled = 32'd1520000; // Mid R, Low GB
            6'b010100: psi_scaled = 32'd1360000; // Mid RG, Low B
            6'b010101: psi_scaled = 32'd1210000; // Mid RGB
            6'b010110: psi_scaled = 32'd1080000; // Mid RG, High B
            6'b011001: psi_scaled = 32'd1180000; // Mid RB, High G
            6'b011010: psi_scaled = 32'd1100000; // Mid R, High GB
            6'b100101: psi_scaled = 32'd1360000; // High R, Mid GB
            6'b100110: psi_scaled = 32'd1020000; // High R, Mid G, High B
            6'b101001: psi_scaled = 32'd1010000; // High RG, Mid B
            6'b101010: psi_scaled = 32'd1170000; // High RGB
            default:   psi_scaled = 32'd1250000; // Default PSI = 1.25
        endcase
    end

endmodule
