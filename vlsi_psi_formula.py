#!/usr/bin/env python3
"""
VLSI-Ready PSI Calculation Module
Optimized mathematical relationship for hardware implementation
Based on analysis of Ar, Ag, Ab -> BestPsi correlation
"""

import numpy as np

class VLSIPsiCalculator:
    """
    Hardware-friendly PSI calculator for VLSI implementation
    """

    def __init__(self):
        # Integer coefficients (scaled by 1,000,000 for precision)
        self.SCALE_FACTOR = 1000000
        self.INTERCEPT = 771580      # 0.771580 * 1,000,000
        self.COEFF_AR = 18641        # 0.018641 * 1,000,000
        self.COEFF_AG = -29403       # -0.029403 * 1,000,000
        self.COEFF_AB = 12765        # 0.012765 * 1,000,000

        # Alternative lookup table approach for ultra-low resource designs
        self.lookup_table = {
            # (Ar_range, Ag_range, Ab_range): BestPsi_scaled
            (0, 0, 0): 1510000,      # Low RGB -> PSI ≈ 1.51
            (0, 0, 1): 1400000,      # Low R,G, Mid B -> PSI ≈ 1.40
            (0, 1, 1): 1240000,      # Low R, Mid G,B -> PSI ≈ 1.24
            (1, 0, 0): 1520000,      # Mid R, Low G,B -> PSI ≈ 1.52
            (1, 1, 0): 1360000,      # Mid R,G, Low B -> PSI ≈ 1.36
            (1, 1, 1): 1210000,      # Mid RGB -> PSI ≈ 1.21
            (1, 1, 2): 1080000,      # Mid R,G, High B -> PSI ≈ 1.08
            (1, 2, 1): 1180000,      # Mid R,B, High G -> PSI ≈ 1.18
            (1, 2, 2): 1100000,      # Mid R, High G,B -> PSI ≈ 1.10
            (2, 1, 1): 1360000,      # High R, Mid G,B -> PSI ≈ 1.36
            (2, 1, 2): 1020000,      # High R, Mid G, High B -> PSI ≈ 1.02
            (2, 2, 1): 1010000,      # High R,G, Mid B -> PSI ≈ 1.01
            (2, 2, 2): 1170000,      # High RGB -> PSI ≈ 1.17
        }

    def calculate_psi_linear(self, ar, ag, ab):
        """
        Calculate PSI using linear formula (most accurate)
        Input: RGB atmospheric light values (0-255)
        Output: PSI value as float
        """
        # Calculate scaled result to avoid floating point
        psi_scaled = (self.INTERCEPT +
                     self.COEFF_AR * ar +
                     self.COEFF_AG * ag +
                     self.COEFF_AB * ab)

        # Convert back to float
        return psi_scaled / self.SCALE_FACTOR

    def calculate_psi_integer(self, ar, ag, ab):
        """
        Calculate PSI using pure integer arithmetic
        Input: RGB atmospheric light values (0-255)
        Output: PSI value scaled by SCALE_FACTOR (divide by 1,000,000 to get actual PSI)
        """
        return (self.INTERCEPT +
                self.COEFF_AR * ar +
                self.COEFF_AG * ag +
                self.COEFF_AB * ab)

    def rgb_to_range(self, value):
        """Convert RGB value to range category for lookup table"""
        if value < 200:
            return 0    # Low
        elif value < 240:
            return 1    # Mid
        else:
            return 2    # High

    def calculate_psi_lookup(self, ar, ag, ab):
        """
        Calculate PSI using lookup table (lowest resource usage)
        Input: RGB atmospheric light values (0-255)
        Output: PSI value as float
        """
        ar_range = self.rgb_to_range(ar)
        ag_range = self.rgb_to_range(ag)
        ab_range = self.rgb_to_range(ab)

        key = (ar_range, ag_range, ab_range)
        psi_scaled = self.lookup_table.get(key, 1250000)  # Default to 1.25 if not found

        return psi_scaled / self.SCALE_FACTOR

    def calculate_psi_optimized(self, ar, ag, ab):
        """
        Optimized calculation using bit shifts and simple operations
        For ultra-low power VLSI implementations
        """
        # Use bit shifts instead of multiplication where possible
        # 18641 ≈ 18432 (18432 = 9 << 11)
        # -29403 ≈ -29440 (29440 = 115 << 8)
        # 12765 ≈ 12800 (12800 = 50 << 8)

        ar_contrib = ar * 18432  # Approximate 18641
        ag_contrib = ag * -29440 # Approximate -29403
        ab_contrib = ab * 12800  # Approximate 12765

        psi_scaled = 771580 + ar_contrib + ag_contrib + ab_contrib

        return psi_scaled / self.SCALE_FACTOR

def generate_verilog_module():
    """Generate Verilog module for VLSI implementation"""

    verilog_code = '''
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
'''

    return verilog_code

def test_implementations():
    """Test different PSI calculation implementations"""

    calc = VLSIPsiCalculator()

    # Test cases from the dataset
    test_cases = [
        (255, 253, 252),  # High RGB
        (253, 246, 220),  # Mixed RGB
        (127, 126, 140),  # Low RGB
        (250, 225, 219),  # Mid-range RGB
    ]

    print("Testing PSI calculation implementations:")
    print("-" * 60)
    print(f"{'Ar':>3} {'Ag':>3} {'Ab':>3} | {'Linear':>8} {'Integer':>10} {'Lookup':>8} {'Optimized':>8}")
    print("-" * 60)

    for ar, ag, ab in test_cases:
        psi_linear = calc.calculate_psi_linear(ar, ag, ab)
        psi_integer = calc.calculate_psi_integer(ar, ag, ab) / calc.SCALE_FACTOR
        psi_lookup = calc.calculate_psi_lookup(ar, ag, ab)
        psi_optimized = calc.calculate_psi_optimized(ar, ag, ab)

        print(f"{ar:>3} {ag:>3} {ab:>3} | {psi_linear:>8.3f} {psi_integer:>10.3f} {psi_lookup:>8.3f} {psi_optimized:>8.3f}")

if __name__ == "__main__":
    # Test the implementations
    test_implementations()

    # Generate Verilog code
    verilog = generate_verilog_module()

    # Save Verilog module
    with open('psi_calculator.v', 'w') as f:
        f.write(verilog)

    print("\nVerilog module saved to 'psi_calculator.v'")
    print("\nVLSI Implementation Summary:")
    print("=" * 50)
    print("1. Linear Formula (Most Accurate):")
    print("   BestPsi = 0.772 + 0.0186*Ar - 0.0294*Ag + 0.0128*Ab")
    print("   Expected R² accuracy: 0.105")

    print("\n2. Integer Formula (Hardware-Friendly):")
    print("   BestPsi_scaled = 771580 + 18641*Ar - 29403*Ag + 12765*Ab")
    print("   BestPsi = BestPsi_scaled / 1,000,000")

    print("\n3. Lookup Table (Ultra-Low Resources):")
    print("   Uses 3-bit quantization of RGB -> Direct PSI lookup")
    print("   Only 64 entries maximum, actual 13 entries used")

    print("\n4. Key Findings for VLSI:")
    print("   - Ar has strongest positive correlation (0.122)")
    print("   - Ag has weak negative correlation (0.027)")
    print("   - Ab has weak positive correlation (0.042)")
    print("   - Overall correlation is weak (R² = 0.105)")
    print("   - Recommend lookup table for best area/power trade-off")