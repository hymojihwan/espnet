#!/usr/bin/env python3
"""JEPA ASR training script."""
from espnet2.tasks.asr_jepa import ASRJEPATask


def get_parser():
    parser = ASRJEPATask.get_parser()
    return parser


def main(cmd=None):
    r"""JEPA ASR training.

    Example:

        % python asr_jepa_train.py asr_jepa --print_config --optim adadelta \
                > conf/train_asr_jepa.yaml
        % python asr_jepa_train.py --config conf/train_asr_jepa.yaml
    """
    ASRJEPATask.main(cmd=cmd)


if __name__ == "__main__":
    main()

