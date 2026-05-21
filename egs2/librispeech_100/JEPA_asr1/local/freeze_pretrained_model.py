#!/usr/bin/env python3
"""Freeze pretrained model parameters except frontend.

This script freezes all model parameters except the frontend
to enable frontend-only training.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add espnet2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "espnet2"))

from espnet2.tasks.asr import ASRTask


def freeze_pretrained_model(model, freeze_frontend=False):
    """Freeze all parameters except frontend.
    
    Args:
        model: ESPnetASRModel instance
        freeze_frontend: If True, also freeze frontend (default: False)
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Freeze all parameters except frontend
        if name.startswith("frontend.") and not freeze_frontend:
            param.requires_grad = True
            trainable_params += param.numel()
            logging.info(f"Trainable: {name} ({param.numel():,} params)")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
            logging.info(f"Frozen: {name} ({param.numel():,} params)")
    
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Frozen parameters: {frozen_params:,}")
    logging.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Build model from config
    # Note: This is a simplified version. In practice, you would use
    # ASRTask.build_model_from_file() or similar
    logging.info("This script should be integrated into the training pipeline")
    logging.info("For now, use the init_param in config file and add freeze logic")


if __name__ == "__main__":
    main()

