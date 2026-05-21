#!/usr/bin/env python3
"""
Add noise to LibriSpeech audio files directly in their original structure.

This script reads LibriSpeech directory structure and adds noise from MUSAN dataset
to each audio file while maintaining the original directory structure.
"""

import os
import sys
import random
import subprocess
import tempfile
import argparse
from pathlib import Path


def add_noise_to_audio(input_path, output_path, musan_dir, snr_min, snr_max, noise_prob, noise_files=None):
    """Add noise to audio file using sox"""
    if random.random() > noise_prob:
        # Copy original file without noise
        subprocess.run(['cp', input_path, output_path])
        return
    
    # Get noise files if not provided
    if noise_files is None:
        noise_files = []
        for root, dirs, files in os.walk(musan_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac'):
                    noise_files.append(os.path.join(root, file))
    
    if not noise_files:
        print(f"Warning: No noise files found in {musan_dir}")
        subprocess.run(['cp', input_path, output_path])
        return
    
    noise_file = random.choice(noise_files)
    snr = random.uniform(snr_min, snr_max)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_noise:
        tmp_noise_path = tmp_noise.name
    
    try:
        # Get input audio duration using sox
        duration_cmd = subprocess.run([
            'sox', '--i', '-D', input_path
        ], capture_output=True, text=True)
        
        if duration_cmd.returncode == 0:
            input_duration = float(duration_cmd.stdout.strip())
        else:
            # Fallback: assume 10 seconds if can't get duration
            input_duration = 10.0
        
        # Convert noise file to match input format and length
        # First trim to 30 seconds if longer, then extend if shorter
        subprocess.run(['sox', noise_file, tmp_noise_path, 'trim', '0', '30'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check if we need to extend the noise file
        noise_duration_cmd = subprocess.run([
            'sox', '--i', '-D', tmp_noise_path
        ], capture_output=True, text=True)
        
        if noise_duration_cmd.returncode == 0:
            actual_noise_duration = float(noise_duration_cmd.stdout.strip())
            if actual_noise_duration < input_duration:
                # Calculate how many repeats we need to exceed the target duration
                repeats_needed = int((input_duration / actual_noise_duration) + 1)  # Add 1 to ensure we exceed target
                
                # Extend noise by repeating it
                extended_noise_path = tmp_noise_path + '_extended'
                repeat_cmd = subprocess.run([
                    'sox', tmp_noise_path, extended_noise_path, 'repeat', str(repeats_needed)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if repeat_cmd.returncode == 0:
                    # Replace the original temp file
                    os.unlink(tmp_noise_path)
                    os.rename(extended_noise_path, tmp_noise_path)
                    
                    # Now trim to exact target duration
                    trimmed_noise_path = tmp_noise_path + '_trimmed'
                    trim_final_cmd = subprocess.run([
                        'sox', tmp_noise_path, trimmed_noise_path, 'trim', '0', str(input_duration)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if trim_final_cmd.returncode == 0:
                        os.unlink(tmp_noise_path)
                        os.rename(trimmed_noise_path, tmp_noise_path)
        
        # Add noise using sox with FLAC format
        subprocess.run([
            'sox', '-m', input_path, 
            f'{tmp_noise_path}:vol={snr}dB', 
            '-r', '16000', '-c', '1', '-b', '16', '-C', '8', output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        # If sox fails, copy original file
        subprocess.run(['cp', input_path, output_path])
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_noise_path):
            os.unlink(tmp_noise_path)


def process_librispeech_directory(input_dir, musan_dir, snr_min, snr_max, noise_prob, seed=42):
    """Process all audio files in LibriSpeech directory structure"""
    random.seed(seed)
    
    # Get noise files once
    noise_files = []
    for root, dirs, files in os.walk(musan_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                noise_files.append(os.path.join(root, file))
    
    if not noise_files:
        print(f"Error: No noise files found in {musan_dir}")
        return False
    
    print(f"Found {len(noise_files)} noise files")
    
    # Process all audio files in the directory structure
    processed_count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                input_path = os.path.join(root, file)
                output_path = input_path  # Overwrite the original file
                
                # Add noise to the audio file
                add_noise_to_audio(input_path, output_path, musan_dir, snr_min, snr_max, noise_prob, noise_files)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
    
    print(f"Total processed files: {processed_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add noise to LibriSpeech audio files")
    parser.add_argument("--input_dir", required=True, help="Input LibriSpeech directory")
    parser.add_argument("--musan_dir", required=True, help="MUSAN noise directory")
    parser.add_argument("--snr_min", type=float, default=-10.0, help="Minimum SNR value")
    parser.add_argument("--snr_max", type=float, default=10.0, help="Maximum SNR value")
    parser.add_argument("--noise_prob", type=float, default=1.0, help="Probability of applying noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate SNR values
    if args.snr_min >= args.snr_max:
        print("Error: snr_min must be less than snr_max")
        return 1
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    # Check if MUSAN directory exists
    if not os.path.isdir(args.musan_dir):
        print(f"Error: MUSAN directory {args.musan_dir} does not exist")
        return 1
    
    # Check if sox is available
    try:
        subprocess.run(['sox', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: sox is not installed. Please install sox first.")
        return 1
    
    print(f"Processing directory: {args.input_dir}")
    print(f"MUSAN directory: {args.musan_dir}")
    print(f"SNR range: {args.snr_min} to {args.snr_max}")
    print(f"Noise probability: {args.noise_prob}")
    print(f"Seed: {args.seed}")
    
    # Process the directory
    success = process_librispeech_directory(
        args.input_dir, args.musan_dir, args.snr_min, args.snr_max, args.noise_prob, args.seed
    )
    
    if success:
        print("Successfully processed all audio files")
        return 0
    else:
        print("Error occurred during processing")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 