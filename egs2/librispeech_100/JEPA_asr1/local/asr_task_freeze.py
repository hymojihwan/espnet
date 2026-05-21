"""Custom ASRTask with freeze functionality for frontend-only training."""

import logging
from pathlib import Path
import sys

# Add espnet2 to path
espnet_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(espnet_root / "espnet2"))

from espnet2.tasks.asr import ASRTask


class ASRTaskFreeze(ASRTask):
    """ASRTask with freeze functionality for frontend-only training."""
    
    @classmethod
    def build_model(cls, args):
        """Build model and freeze all parameters except frontend."""
        model = super().build_model(args)
        
        # Freeze all parameters except frontend
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            # Keep frontend trainable, freeze everything else
            if name.startswith("frontend."):
                param.requires_grad = True
                trainable_params += param.numel()
                logging.info(f"Trainable: {name} ({param.numel():,} params)")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        logging.info("=" * 60)
        logging.info("Model Parameter Freeze Summary:")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        logging.info(f"  Frozen parameters: {frozen_params:,}")
        logging.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
        logging.info("=" * 60)
        
        return model

