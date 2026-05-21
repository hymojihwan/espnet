#!/usr/bin/env python3
"""SE-JEPA ASR training script.

Flow: SE(wav, SI-SNR) → enhanced_wav → log-mel → JEPA predictive → Conformer → CTC
"""

from espnet2.tasks.asr_se_jepa import ASRSEJEPATask


def get_parser():
    return ASRSEJEPATask.get_parser()


def main(cmd=None):
    r"""SE-JEPA ASR training.

    Example:
        % python asr_se_jepa_train.py asr_se_jepa --print_config
        % python asr_se_jepa_train.py --config conf/train_se_jepa.yaml
    """
    ASRSEJEPATask.main(cmd=cmd)


if __name__ == "__main__":
    main()
