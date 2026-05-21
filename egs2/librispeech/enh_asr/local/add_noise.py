#!/usr/bin/env python3
"""
Add noise to audio files using MUSAN dataset.

This script reads a wav.scp file and adds noise from MUSAN dataset to each audio file.
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
    # Always apply noise - remove probability check
    # if random.random() > noise_prob:
    #     # Copy original file without noise
    #     print(f"Copying original file (no noise applied): {input_path} -> {output_path}")
    #     subprocess.run(['cp', input_path, output_path])
    #     return
    
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
    # Adjust SNR to make noise more audible
    # Lower SNR means more noise, higher SNR means less noise
    # For better noise effect, use lower SNR values
    snr = random.uniform(snr_min, snr_max)
    
    print(f"Adding noise to {input_path} using {noise_file} with SNR {snr:.2f}dB")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_noise:
        tmp_noise_path = tmp_noise.name
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_noise_vol:
        tmp_noise_vol_path = tmp_noise.name
    
    try:
        # Get input audio duration using sox
        duration_cmd = subprocess.run([
            'sox', '--i', '-D', input_path
        ], capture_output=True, text=True)
        
        if duration_cmd.returncode == 0:
            input_duration = float(duration_cmd.stdout.strip())
            print(f"Input audio duration: {input_duration}s")
        else:
            # Fallback: assume 10 seconds if can't get duration
            input_duration = 10.0
            print(f"Could not get duration, using fallback: {input_duration}s")
            print(f"Duration command error: {duration_cmd.stderr}")
        
        # Convert noise file to match input format and length
        # If noise is shorter than input, repeat it; if longer, trim it
        trim_cmd = subprocess.run([
            'sox', noise_file, tmp_noise_path, 
            'trim', '0', str(input_duration)
        ], capture_output=True, text=True)
        
        if trim_cmd.returncode != 0:
            print(f"Error trimming noise file: {trim_cmd.stderr}")
            raise Exception("Failed to trim noise file")
        
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
                ], capture_output=True, text=True)
                
                if repeat_cmd.returncode == 0:
                    # Replace the original temp file
                    os.unlink(tmp_noise_path)
                    os.rename(extended_noise_path, tmp_noise_path)
                    print(f"Extended noise from {actual_noise_duration}s to exceed {input_duration}s using {repeats_needed} repeats")
                    
                    # Now trim to exact target duration
                    trimmed_noise_path = tmp_noise_path + '_trimmed'
                    trim_final_cmd = subprocess.run([
                        'sox', tmp_noise_path, trimmed_noise_path, 'trim', '0', str(input_duration)
                    ], capture_output=True, text=True)
                    
                    if trim_final_cmd.returncode == 0:
                        os.unlink(tmp_noise_path)
                        os.rename(trimmed_noise_path, tmp_noise_path)
                        print(f"Trimmed noise to exact {input_duration}s")
                    else:
                        print(f"Warning: Could not trim noise to exact duration: {trim_final_cmd.stderr}")
                else:
                    print(f"Warning: Could not extend noise file: {repeat_cmd.stderr}")
        
        print(f"Successfully prepared noise file to match {input_duration}s")
        
        # Adjust noise volume based on SNR
        # For better noise effect, we'll use a more aggressive approach
        # Convert SNR to volume adjustment: positive SNR means noise is quieter
        # We want the noise to be more audible, so we'll make it louder
        volume_adjustment = -snr  # Negative because sox vol expects negative values for louder
        vol_cmd = subprocess.run([
            'sox', tmp_noise_path, tmp_noise_vol_path, 'vol', f'{volume_adjustment}dB'
        ], capture_output=True, text=True)
        
        if vol_cmd.returncode != 0:
            print(f"Error adjusting noise volume: {vol_cmd.stderr}")
            raise Exception("Failed to adjust noise volume")
        
        print(f"Successfully adjusted noise volume to {volume_adjustment}dB (SNR: {snr}dB)")
        
        # Add noise using sox with FLAC format
        # Use simple mixing without volume effects in the input specification
        mix_cmd = subprocess.run([
            'sox', '-m', 
            input_path,  # Original audio
            tmp_noise_vol_path,  # Noise with adjusted volume
            '-r', '16000', '-c', '1', '-b', '16', '-C', '8', output_path
        ], capture_output=True, text=True)
        
        if mix_cmd.returncode != 0:
            print(f"Error mixing audio: {mix_cmd.stderr}")
            raise Exception("Failed to mix audio")
        
        print(f"Successfully created noisy audio: {output_path}")
        
        # Verify the output file exists and has different size/content
        if os.path.exists(output_path):
            original_size = os.path.getsize(input_path)
            noisy_size = os.path.getsize(output_path)
            print(f"Original: {original_size} bytes, Noisy: {noisy_size} bytes")
            
            # Check if files are actually different (basic check)
            if original_size == noisy_size:
                print(f"Warning: Output file size same as input - noise may not have been added properly")
        else:
            print(f"Warning: Output file was not created")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        # If sox fails, copy original file
        print(f"Copying original file due to error: {input_path} -> {output_path}")
        subprocess.run(['cp', input_path, output_path])
    finally:
        # Clean up temporary files
        for tmp_path in [tmp_noise_path, tmp_noise_vol_path]:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def process_wav_scp(input_scp, output_scp, musan_dir, snr_min, snr_max, noise_prob, output_dir, dataset_name=None):
    """Process wav.scp file and add noise to audio files"""
    if not os.path.exists(input_scp):
        print(f"Error: {input_scp} does not exist")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all noise files from MUSAN
    all_noise_files = []
    for root, dirs, files in os.walk(musan_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                all_noise_files.append(os.path.join(root, file))
    
    if not all_noise_files:
        print(f"Error: No noise files found in {musan_dir}")
        return False
    
    # Select different noise files for each dataset
    if dataset_name:
        # Use dataset name to create different noise file subsets
        dataset_hash = hash(dataset_name) % len(all_noise_files)
        start_idx = (dataset_hash * 1000) % len(all_noise_files)
        end_idx = min(start_idx + len(all_noise_files) // 4, len(all_noise_files))
        noise_files = all_noise_files[start_idx:end_idx]
        
        # If we don't have enough files, wrap around
        if len(noise_files) < 100:
            noise_files = all_noise_files
        
        print(f"Dataset '{dataset_name}': Using {len(noise_files)} noise files out of {len(all_noise_files)} total")
    else:
        noise_files = all_noise_files
    
    with open(input_scp, 'r') as f_in, open(output_scp, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            
            utt_id, audio_path = parts
            
            # Parse utterance ID to get speaker and chapter info
            # Format: {speaker}-{chapter}-{utterance}
            # Example: 1272-128104-0000
            utt_parts = utt_id.split('-')
            if len(utt_parts) >= 3:
                speaker = utt_parts[0]
                chapter = utt_parts[1]
                utterance = '-'.join(utt_parts[2:])
                
                # Create directory structure: noisy_librispeech/{dataset_name}/{speaker}/{chapter}/
                # This maintains the original LibriSpeech structure with dataset name
                if dataset_name:
                    dataset_dir = os.path.join(output_dir, dataset_name)
                    speaker_dir = os.path.join(dataset_dir, speaker)
                else:
                    speaker_dir = os.path.join(output_dir, speaker)
                
                chapter_dir = os.path.join(speaker_dir, chapter)
                os.makedirs(chapter_dir, exist_ok=True)
                
                # Generate output path maintaining original structure
                output_filename = f"{utt_id}.flac"
                output_path = os.path.join(chapter_dir, output_filename)
            else:
                # Fallback to flat structure if utterance ID format is unexpected
                if dataset_name:
                    dataset_dir = os.path.join(output_dir, dataset_name)
                    os.makedirs(dataset_dir, exist_ok=True)
                    output_filename = f"{utt_id}.flac"
                    output_path = os.path.join(dataset_dir, output_filename)
                else:
                    output_filename = f"{utt_id}.flac"
                    output_path = os.path.join(output_dir, output_filename)
            
            # Add noise to audio with dataset-specific noise files
            add_noise_to_audio(audio_path, output_path, musan_dir, snr_min, snr_max, noise_prob, noise_files)
            
            # Write to wav.scp with absolute path for noisy data
            f_out.write(f"{utt_id} {output_path}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Add noise to audio files using MUSAN dataset")
    parser.add_argument("--input_scp", required=True, help="Input wav.scp file")
    parser.add_argument("--output_scp", required=True, help="Output wav.scp file")
    parser.add_argument("--musan_dir", required=True, help="MUSAN dataset directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for noisy audio files")
    parser.add_argument("--snr_min", type=float, default=-10.0, help="Minimum SNR (default: -10.0)")
    parser.add_argument("--snr_max", type=float, default=10.0, help="Maximum SNR (default: 10.0)")
    parser.add_argument("--noise_prob", type=float, default=1.0, help="Probability of applying noise (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name for noise file selection (default: None)")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Check if MUSAN directory exists
    if not os.path.exists(args.musan_dir):
        print(f"Error: MUSAN directory {args.musan_dir} does not exist")
        sys.exit(1)
    
    # Process wav.scp
    success = process_wav_scp(
        args.input_scp, 
        args.output_scp, 
        args.musan_dir, 
        args.snr_min, 
        args.snr_max, 
        args.noise_prob,
        args.output_dir,
        args.dataset_name
    )
    
    if success:
        print(f"Successfully created noisy wav.scp: {args.output_scp}")
    else:
        print("Failed to process wav.scp")
        sys.exit(1)


if __name__ == "__main__":
    main() 