#!/usr/bin/env python3
"""
Accurate noise addition using RMS-based SNR calculation
"""

import os
import sys
import random
import glob
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

def cal_rms(amp):
    """Calculate RMS of audio signal"""
    return np.sqrt(np.mean(amp ** 2))

def cal_adjusted_rms(clean_rms, snr):
    """Calculate required noise RMS for target SNR"""
    return clean_rms / (10 ** (snr / 20.0))

def save_waveform(file_path, sample_rate, amp):
    """Save audio file as FLAC"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Convert back to int16 for saving
    amp_int16 = np.clip(amp, -32768, 32767).astype(np.int16)
    # Save as FLAC format
    sf.write(file_path, amp_int16, sample_rate, format='FLAC', subtype='PCM_16')

def add_noise_accurate(clean_file, noise_file, output_file, snr_min=-10.0, snr_max=10.0):
    """Add noise with accurate SNR calculation"""
    
    # Load clean audio
    clean_amp, sample_rate = sf.read(clean_file, dtype='int16')
    clean_amp = clean_amp.astype(np.float64)
    clean_rms = cal_rms(clean_amp)
    
    # Load noise audio
    noise_amp, noise_rate = sf.read(noise_file, dtype='int16')
    noise_amp = noise_amp.astype(np.float64)
    
    # Resample noise if needed
    if noise_rate != sample_rate:
        noise_amp = sf.resample(noise_amp, sample_rate, noise_rate)
    
    # Adjust noise length to match clean audio
    if len(noise_amp) < len(clean_amp):
        repeat_factor = int(np.ceil(len(clean_amp) / len(noise_amp)))
        noise_amp = np.tile(noise_amp, repeat_factor)
    
    # Select random segment from noise
    start = random.randint(0, len(noise_amp) - len(clean_amp))
    selected_noise = noise_amp[start:start + len(clean_amp)]
    noise_rms = cal_rms(selected_noise)
    
    # Apply SNR
    snr = random.uniform(snr_min, snr_max)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    adjusted_noise = selected_noise * (adjusted_noise_rms / noise_rms)
    
    # Mix clean and noise
    mixed_amp = clean_amp + adjusted_noise
    
    # Prevent clipping
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed_amp.max() > max_int16 or mixed_amp.min() < min_int16:
        reduction_rate = min(max_int16 / mixed_amp.max(), min_int16 / mixed_amp.min())
        mixed_amp *= reduction_rate
    
    # Save
    save_waveform(output_file, sample_rate, mixed_amp)
    
    return snr

def process_librispeech(clean_dir, noise_dir, noisy_dir, snr_min=-10.0, snr_max=10.0):
    """Process specified LibriSpeech datasets"""
    
    # Define datasets to process
    datasets = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100"]
    
    print(f"Processing LibriSpeech from {clean_dir}")
    print(f"Adding noise from {noise_dir}")
    print(f"Output to {noisy_dir}")
    print(f"SNR range: {snr_min} to {snr_max} dB")
    print(f"Target datasets: {', '.join(datasets)}")
    
    # Get all noise files
    all_noise_files = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
    if not all_noise_files:
        raise ValueError(f"No noise files found in {noise_dir}")
    
    print(f"Found {len(all_noise_files)} total noise files")
    
    # Split noise files by dataset type
    # train: 60%, dev: 20%, test: 20%
    random.shuffle(all_noise_files)  # Shuffle all noise files first
    
    total_noise = len(all_noise_files)
    train_noise_count = int(total_noise * 0.6)
    dev_noise_count = int(total_noise * 0.2)
    test_noise_count = total_noise - train_noise_count - dev_noise_count
    
    train_noise_files = all_noise_files[:train_noise_count]
    dev_noise_files = all_noise_files[train_noise_count:train_noise_count + dev_noise_count]
    test_noise_files = all_noise_files[train_noise_count + dev_noise_count:]
    
    print(f"Noise file distribution:")
    print(f"  Train datasets: {len(train_noise_files)} files ({len(train_noise_files)/total_noise*100:.1f}%)")
    print(f"  Dev datasets: {len(dev_noise_files)} files ({len(dev_noise_files)/total_noise*100:.1f}%)")
    print(f"  Test datasets: {len(test_noise_files)} files ({len(test_noise_files)/total_noise*100:.1f}%)")
    
    # Process each dataset separately with different noise pools
    total_processed = 0
    total_failed = 0
    
    for dataset in datasets:
        dataset_path = os.path.join(clean_dir, dataset)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset {dataset} not found, skipping")
            continue
            
        print(f"\n--- Processing {dataset} ---")
        
        # Get audio files for this dataset
        dataset_files = glob.glob(os.path.join(dataset_path, "**", "*.flac"), recursive=True)
        if not dataset_files:
            print(f"No audio files found in {dataset}, skipping")
            continue
            
        print(f"Found {len(dataset_files)} audio files in {dataset}")
        
        # Get text files for this dataset
        text_files = glob.glob(os.path.join(dataset_path, "**", "*.trans.txt"), recursive=True)
        print(f"Found {len(text_files)} text files in {dataset}")
        
        # Select appropriate noise pool based on dataset type
        if "train" in dataset:
            noise_files = train_noise_files
            print(f"  Using train noise pool: {len(noise_files)} files")
        elif "dev" in dataset:
            noise_files = dev_noise_files
            print(f"  Using dev noise pool: {len(noise_files)} files")
        else:  # test datasets
            noise_files = test_noise_files
            print(f"  Using test noise pool: {len(noise_files)} files")
        
        # Shuffle noise files for this dataset
        random.shuffle(noise_files)
        
        processed = 0
        failed = 0
        
        # First, copy all text files
        print(f"  Copying text files...")
        for text_file in text_files:
            try:
                relative_path = os.path.relpath(text_file, clean_dir)
                noisy_text_path = os.path.join(noisy_dir, relative_path)
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(noisy_text_path), exist_ok=True)
                
                # Copy text file
                import shutil
                shutil.copy2(text_file, noisy_text_path)
                
            except Exception as e:
                print(f"✗ Failed to copy text file {text_file}: {e}")
        
        # Then process audio files
        print(f"  Processing audio files...")
        for clean_file in dataset_files:
            try:
                # Select noise file from this dataset's noise pool
                noise_file = noise_files[processed % len(noise_files)]
                
                # Create output path
                relative_path = os.path.relpath(clean_file, clean_dir)
                noisy_file_path = os.path.join(noisy_dir, relative_path)
                
                # Add noise
                snr = add_noise_accurate(clean_file, noise_file, noisy_file_path, snr_min, snr_max)
                
                processed += 1
                
                # Log progress for this dataset
                if processed % 100 == 0:
                    progress_percent = (processed / len(dataset_files)) * 100
                    print(f"  {dataset}: {processed}/{len(dataset_files)} audio files processed ({progress_percent:.1f}%)")
                    
            except Exception as e:
                print(f"✗ Failed to process {clean_file}: {e}")
                failed += 1
        
        print(f"  {dataset}: Completed - {processed} audio files processed, {failed} failed")
        total_processed += processed
        total_failed += failed
    
    print(f"\n=== Final Summary ===")
    print(f"Total completed: {total_processed} audio files processed, {total_failed} failed")
    print(f"Text files copied to noisy directory")
    return total_processed, total_failed

def main():
    parser = argparse.ArgumentParser(description="Add noise to LibriSpeech with accurate SNR")
    parser.add_argument("--clean_dir", default="/DB/LibriSpeech", help="Clean LibriSpeech directory")
    parser.add_argument("--noise_dir", default="/DB/musan", help="MUSAN noise directory")
    parser.add_argument("--noisy_dir", default="/DB/noisy_librispeech", help="Output directory")
    parser.add_argument("--snr_min", type=float, default=-10.0, help="Minimum SNR (default: 0.0)")
    parser.add_argument("--snr_max", type=float, default=10.0, help="Maximum SNR (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Check directories
    if not os.path.exists(args.clean_dir):
        print(f"Error: Clean directory {args.clean_dir} does not exist")
        return 1
    
    if not os.path.exists(args.noise_dir):
        print(f"Error: Noise directory {args.noise_dir} does not exist")
        return 1
    
    # Process files
    try:
        processed, failed = process_librispeech(
            args.clean_dir, args.noise_dir, args.noisy_dir, args.snr_min, args.snr_max
        )
        
        if failed == 0:
            print("✓ All files processed successfully!")
            return 0
        else:
            print(f"⚠ {failed} files failed to process")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 