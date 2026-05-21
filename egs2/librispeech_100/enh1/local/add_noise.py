#!/usr/bin/env python3
"""Create noisy data by adding noise to clean speech."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def add_noise(clean_wav, noise_wav, snr_db):
    """Add noise to clean speech at specified SNR."""
    # Calculate signal power
    clean_power = np.mean(clean_wav ** 2)
    noise_power = np.mean(noise_wav ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-10:
        noise_power = 1e-10
    
    # Calculate noise scaling factor
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(clean_power / (snr_linear * noise_power))
    
    # Add noise
    noisy_wav = clean_wav + noise_scale * noise_wav
    
    return noisy_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_scp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_scp", type=str, required=True)
    parser.add_argument("--min_snr", type=float, default=0)
    parser.add_argument("--max_snr", type=float, default=20)
    parser.add_argument("--fixed_snr", type=float, default=None, 
                        help="Use fixed SNR instead of random")
    parser.add_argument("--random_snr", action="store_true",
                        help="Use random SNR between min_snr and max_snr")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process clean speech
    output_scp = []
    with open(args.clean_scp, "r") as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Adding noise"):
        utt_id, clean_path = line.strip().split(None, 1)
        
        # Read clean speech
        clean_wav, sr = sf.read(clean_path)
        if sr != args.sample_rate:
            print(f"Warning: sample rate mismatch for {utt_id}: {sr} != {args.sample_rate}")
        
        # Generate white noise
        noise_wav = np.random.randn(len(clean_wav))
        
        # Determine SNR
        if args.fixed_snr is not None:
            # Use fixed SNR for evaluation
            snr_db = args.fixed_snr
        elif args.random_snr:
            # Use random SNR for training
            snr_db = random.uniform(args.min_snr, args.max_snr)
        else:
            # Default: use max_snr
            snr_db = args.max_snr
        
        # Add noise
        noisy_wav = add_noise(clean_wav, noise_wav, snr_db)
        
        # Normalize to prevent clipping
        max_val = np.abs(noisy_wav).max()
        if max_val > 0.99:
            noisy_wav = noisy_wav * 0.99 / max_val
        
        # Save noisy speech
        output_path = os.path.join(args.output_dir, f"{utt_id}.wav")
        sf.write(output_path, noisy_wav, args.sample_rate)
        
        output_scp.append(f"{utt_id} {output_path}\n")
    
    # Write output scp
    with open(args.output_scp, "w") as f:
        f.writelines(output_scp)
    
    if args.fixed_snr is not None:
        print(f"Created {len(output_scp)} noisy files with fixed SNR={args.fixed_snr}dB in {args.output_dir}")
    elif args.random_snr:
        print(f"Created {len(output_scp)} noisy files with random SNR=[{args.min_snr}, {args.max_snr}]dB in {args.output_dir}")
    else:
        print(f"Created {len(output_scp)} noisy files in {args.output_dir}")


if __name__ == "__main__":
    main()

