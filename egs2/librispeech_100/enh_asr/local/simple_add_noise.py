#!/usr/bin/env python3
"""
Simple script to add MUSAN noise to LibriSpeech clean files
"""

import os
import sys
import random
import subprocess
import argparse
from pathlib import Path

def add_noise_simple(input_file, output_file, musan_dir, snr_min=-5, snr_max=5):
    """Add noise to audio file with simple sox mixing"""
    
    # Find random noise file from MUSAN
    noise_files = []
    for root, dirs, files in os.walk(musan_dir):
        for file in files:
            if file.endswith(('.wav', '.flac')):
                noise_files.append(os.path.join(root, file))
    
    if not noise_files:
        print(f"No noise files found in {musan_dir}")
        return False
    
    noise_file = random.choice(noise_files)
    
    # Random SNR between min and max
    snr = random.uniform(snr_min, snr_max)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create temporary noise file with adjusted volume
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_noise:
        tmp_noise_path = tmp_noise.name
    
    try:
        # Adjust noise volume based on SNR
        # Positive SNR = noise is quieter, Negative SNR = noise is louder
        vol_adjustment = -snr  # Inverse relationship
        
        # Apply volume adjustment to noise file
        vol_cmd = [
            'sox', noise_file, tmp_noise_path,
            'vol', str(vol_adjustment), 'dB'
        ]
        
        vol_result = subprocess.run(vol_cmd, capture_output=True, text=True)
        if vol_result.returncode != 0:
            print(f"✗ Failed to adjust noise volume: {vol_result.stderr}")
            return False
        
        # Mix original audio with adjusted noise
        mix_cmd = [
            'sox', '-m',
            input_file,  # Original audio
            tmp_noise_path,  # Noise with adjusted volume
            '-r', '16000', '-c', '1', '-b', '16', output_file
        ]
        
        mix_result = subprocess.run(mix_cmd, capture_output=True, text=True)
        if mix_result.returncode == 0:
            print(f"✓ Added noise to {os.path.basename(input_file)} (SNR: {snr:.1f}dB, vol: {vol_adjustment:.1f}dB)")
            return True
        else:
            print(f"✗ Failed to mix audio: {mix_result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_noise_path):
            os.unlink(tmp_noise_path)

def process_librispeech(input_dir, output_dir, musan_dir, snr_min=-5, snr_max=5):
    """Process all LibriSpeech files"""
    
    print(f"Processing LibriSpeech from {input_dir}")
    print(f"Adding noise from {musan_dir}")
    print(f"Output to {output_dir}")
    print(f"SNR range: {snr_min} to {snr_max} dB")
    
    processed = 0
    failed = 0
    
    # Walk through all LibriSpeech files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                input_path = os.path.join(root, file)
                
                # Create relative path for output
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Add noise
                if add_noise_simple(input_path, output_path, musan_dir, snr_min, snr_max):
                    processed += 1
                else:
                    failed += 1
                
                if (processed + failed) % 100 == 0:
                    print(f"Progress: {processed + failed} files processed")
    
    print(f"\nCompleted: {processed} files processed, {failed} failed")
    return processed, failed

def main():
    parser = argparse.ArgumentParser(description="Add MUSAN noise to LibriSpeech")
    parser.add_argument("--input", default="/DB/LibriSpeech", help="Input LibriSpeech directory")
    parser.add_argument("--output", default="/DB/noisy_librispeech", help="Output directory")
    parser.add_argument("--musan", default="/DB/musan", help="MUSAN noise directory")
    parser.add_argument("--snr_min", type=float, default=-10.0, help="Minimum SNR (default: -5.0)")
    parser.add_argument("--snr_max", type=float, default=10.0, help="Maximum SNR (default: 5.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Check directories
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        return 1
    
    if not os.path.exists(args.musan):
        print(f"Error: MUSAN directory {args.musan} does not exist")
        return 1
    
    # Process files
    processed, failed = process_librispeech(
        args.input, args.output, args.musan, args.snr_min, args.snr_max
    )
    
    if failed == 0:
        print("✓ All files processed successfully!")
        return 0
    else:
        print(f"⚠ {failed} files failed to process")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 