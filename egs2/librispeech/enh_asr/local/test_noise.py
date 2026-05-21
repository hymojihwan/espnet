#!/usr/bin/env python3
"""
Test script to verify noise addition functionality
"""

import os
import sys
import random
import subprocess
import tempfile

def test_noise_addition():
    """Test noise addition with a simple audio file"""
    
    # Check if sox is available
    try:
        subprocess.run(['sox', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ sox is available")
    except FileNotFoundError:
        print("✗ sox is not installed")
        return False
    
    # Check MUSAN directory
    musan_dir = "/DB/musan"
    if not os.path.exists(musan_dir):
        print(f"✗ MUSAN directory {musan_dir} does not exist")
        return False
    
    # Find noise files
    noise_files = []
    for root, dirs, files in os.walk(musan_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                noise_files.append(os.path.join(root, file))
                if len(noise_files) >= 10:  # Just get first 10 files
                    break
        if len(noise_files) >= 10:
            break
    
    if not noise_files:
        print("✗ No noise files found in MUSAN directory")
        return False
    
    print(f"✓ Found {len(noise_files)} noise files")
    
    # Create a test audio file (1 second of silence)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        test_audio_path = tmp_audio.name
    
    try:
        # Generate 1 second of silence
        subprocess.run([
            'sox', '-n', '-r', '16000', '-c', '1', test_audio_path, 'trim', '0.0', '1.0'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"✓ Created test audio file: {test_audio_path}")
        
        # Test noise addition
        output_path = test_audio_path.replace('.wav', '_noisy.wav')
        
        # Import and test the noise function
        sys.path.append('.')
        from add_noise import add_noise_to_audio
        
        print("Testing noise addition...")
        add_noise_to_audio(
            test_audio_path, 
            output_path, 
            musan_dir, 
            snr_min=5.0, 
            snr_max=15.0, 
            noise_prob=1.0  # This parameter is now ignored - always applies noise
        )
        
        if os.path.exists(output_path):
            print(f"✓ Successfully created noisy audio: {output_path}")
            
            # Check file sizes
            original_size = os.path.getsize(test_audio_path)
            noisy_size = os.path.getsize(output_path)
            print(f"Original file size: {original_size} bytes")
            print(f"Noisy file size: {noisy_size} bytes")
            
            # Clean up
            os.unlink(test_audio_path)
            os.unlink(output_path)
            
            return True
        else:
            print("✗ Failed to create noisy audio file")
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {str(e)}")
        return False
    finally:
        # Clean up test files
        for path in [test_audio_path, output_path]:
            if os.path.exists(path):
                os.unlink(path)

if __name__ == "__main__":
    print("Testing noise addition functionality...")
    success = test_noise_addition()
    
    if success:
        print("\n✓ All tests passed! Noise addition is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Tests failed! There are issues with noise addition.")
        sys.exit(1) 