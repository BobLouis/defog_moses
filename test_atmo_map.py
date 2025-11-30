import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from defog_proposed_atmo_section import get_section_atmo_map

def test_atmo_map():
    # Create a synthetic image (1000x1000)
    h, w = 1000, 1000
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Top half: Dark (should have low A?) 
    # Actually A is estimated from the brightest pixel in the dark channel.
    # Let's make different sections have different "brightest pixels" in their dark regions.
    
    # Section 0 (0-50): A should be [100, 100, 100]
    img[0:50, :, :] = 50
    img[0:20, 0:20, :] = 100 # Bright patch (20x20) to survive min filter (size 8)
    
    # Section 10 (500-550): A should be [200, 200, 200]
    img[500:550, :, :] = 50
    img[500:520, 0:20, :] = 200 # Bright patch (20x20)
    
    print("Testing get_section_atmo_map...")
    a_map, a_values = get_section_atmo_map(img, section_count=20, padding_length=50)
    
    print(f"A Map Shape: {a_map.shape}")
    print(f"A Values List Length: {len(a_values)}")
    
    # Check shape
    assert a_map.shape == (h, w, 3)
    
    # Check values in core regions
    # Section 0 core (e.g., row 25)
    a_sec0 = a_map[25, 0, :]
    print(f"A at section 0: {a_sec0}")
    # Note: The estimation logic finds the brightest pixel in the dark channel.
    # If the whole patch is 50, dark channel is 50. Brightest pixel is 100.
    # So A should be [100, 100, 100].
    
    # Section 10 core (e.g., row 525)
    a_sec10 = a_map[525, 0, :]
    print(f"A at section 10: {a_sec10}")
    
    assert np.allclose(a_sec0, [100, 100, 100], atol=5)
    assert np.allclose(a_sec10, [200, 200, 200], atol=5)
    
    # Check transition
    # Boundary between Section 9 and 10 is at 500.
    # Padding is 50, so transition is 475 to 525.
    # Wait, Section 9 is 450-500. Section 10 is 500-550.
    # Section 9 A value? We didn't set it explicitly, it defaults to 255 if empty or 50 if background.
    # Let's check the transition between two known sections if possible, or just check smoothness.
    
    transition_slice = a_map[475:525, 0, 0] # Red channel
    diffs = np.diff(transition_slice)
    
    print(f"Mean diff in transition: {np.mean(diffs):.5f}")
    print(f"Std dev of diffs: {np.std(diffs):.5f}")
    
    # It should be linear, so std dev of diffs should be small
    assert np.std(diffs) < 1e-3, "Transition is not linear!"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_atmo_map()
