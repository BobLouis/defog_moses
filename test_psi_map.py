import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from defog_proposed_psi_fog_esti_corr_clean_section import get_section_psi_map

def test_psi_map():
    # Create a synthetic image (1000x1000)
    # Make it have different characteristics in different sections to force different Psi values
    h, w = 1000, 1000
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Top half: dark (high fog density -> high psi)
    # Bottom half: bright/saturated (low fog density -> low psi)
    img[:500, :, :] = 50 # Dark gray
    img[500:, :, :] = 200 # Light gray
    # Add some color to affect saturation
    img[500:, :, 0] = 255 # Reddish
    
    print("Testing get_section_psi_map...")
    psi_map, psi_values = get_section_psi_map(img, section_count=20, padding_length=50)
    
    print(f"Psi Map Shape: {psi_map.shape}")
    print(f"Psi Values List Length: {len(psi_values)}")
    print(f"Psi Values: {[f'{x:.3f}' for x in psi_values]}")
    
    # Check shape
    assert psi_map.shape == (h, w)
    
    # Check values range
    assert np.all(psi_map >= 0.7)
    assert np.all(psi_map <= 1.3)
    
    # Check transition
    # Section 9 (450-500) and Section 10 (500-550) should have different values
    # Boundary is at 500. Padding is 50, so transition is 475 to 525.
    
    val_470 = psi_map[470, 0] # Should be close to Psi[9]
    val_530 = psi_map[530, 0] # Should be close to Psi[10]
    val_500 = psi_map[500, 0] # Should be approx average
    
    print(f"Value at 470 (Section 9 core): {val_470:.3f}")
    print(f"Value at 500 (Boundary): {val_500:.3f}")
    print(f"Value at 530 (Section 10 core): {val_530:.3f}")
    
    assert np.isclose(val_470, psi_values[9], atol=1e-5)
    assert np.isclose(val_530, psi_values[10], atol=1e-5)
    
    # Check smoothness
    # The derivative in the transition zone should be constant (linear ramp)
    transition_slice = psi_map[475:525, 0]
    diffs = np.diff(transition_slice)
    
    # Diffs should be roughly constant
    print(f"Mean diff in transition: {np.mean(diffs):.5f}")
    print(f"Std dev of diffs: {np.std(diffs):.5f}")
    
    assert np.std(diffs) < 1e-5, "Transition is not linear!"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_psi_map()
