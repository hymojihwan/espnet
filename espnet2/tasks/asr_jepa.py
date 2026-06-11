"""ASR Task for JEPA (Joint Embedding Predictive Architecture) Frontend.

This task extends ASRTask to support JEPA frontend training with clean speech input.
"""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.espnet_jepa_model import ESPnetJEPAASRModel
from espnet2.asr.espnet_se_jepa_model import ESPnetSEJEPAASRModel
from espnet2.asr.frontend.jepa_residual import JEPAResidualFrontend
from espnet2.asr.frontend.jepa_masked import JEPA_MaskedPatchFrontend
from espnet2.asr.frontend.jepa_masked_feature import JEPA_MaskedPatchFeatureFrontend
from espnet2.asr.frontend.jepa_Vit import JEPA_MaskedPatchFrontend as JEPA_ViTFrontend
from espnet2.asr.frontend.jepa_hybrid import JEPA_HybridFrontend
from espnet2.asr.frontend.jepa_audio import JEPA_MaskedPatchLatentFrontend
from espnet2.asr.frontend.jepa_balanced import JEPA_BalancedFrontend
from espnet2.asr.frontend.jepa_mel_latent import JEPAMelLatentFrontend
from espnet2.asr.frontend.frozen_enh import FrozenEnhFrontend
from espnet2.asr.frontend.se_jepa import SE_JEPAFrontend
from espnet2.tasks.asr import (
    ASRTask,
    decoder_choices,
    encoder_choices,
    frontend_choices as base_frontend_choices,
    normalize_choices,
    postencoder_choices,
    preencoder_choices,
    preprocessor_choices,
    specaug_choices,
)
from espnet2.train.class_choices import ClassChoices
from espnet2.train.abs_espnet_model import AbsESPnetModel

# Extend frontend_choices with JEPA frontends
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        **base_frontend_choices.classes,
        jepa_residual=JEPAResidualFrontend,
        jepa_masked=JEPA_MaskedPatchFrontend,
        jepa_masked_feature=JEPA_MaskedPatchFeatureFrontend,
        jepa_vit=JEPA_ViTFrontend,
        jepa_hybrid=JEPA_HybridFrontend,
        jepa_audio=JEPA_MaskedPatchLatentFrontend,
        jepa_balanced=JEPA_BalancedFrontend,
        jepa_mel_latent=JEPAMelLatentFrontend,
        se_jepa=SE_JEPAFrontend,
    ),
    type_check=base_frontend_choices.base_type,
    default=base_frontend_choices.default,
)

# Create JEPA-specific model choices
jepa_model_choices = ClassChoices(
    "model",
    classes=dict(
        jepa_espnet=ESPnetJEPAASRModel,
        se_jepa_espnet=ESPnetSEJEPAASRModel,
    ),
    type_check=AbsESPnetModel,
    default="jepa_espnet",
)


class ASRJEPATask(ASRTask):
    """ASR Task for JEPA Frontend Training.

    This task extends ASRTask to support JEPA frontend training
    by accepting clean_speech input for computing target embeddings.
    """

    # Override class_choices_list to use JEPA model choices
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --model and --model_conf (use JEPA model choices)
        jepa_model_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    feature_domain_frontends = {"jepa_masked_feature"}

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Return required data names for JEPA training.

        Args:
            train: Training mode flag
            inference: Inference mode flag

        Returns:
            Required data names tuple
        """
        if not inference:
            # Training:
            # - SE/ASR (e.g., frozen_enh frontend) only needs speech+text
            # - JEPA variants can consume clean_speech when provided
            retval = ("speech", "text")
        else:
            # Inference: only need speech (noisy)
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Return optional data names for JEPA training.

        Args:
            train: Training mode flag
            inference: Inference mode flag

        Returns:
            Optional data names tuple
        """
        MAX_REFERENCE_NUM = 4

        retval = ["clean_speech"]
        retval += ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval = retval + ["prompt"]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval}")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetJEPAASRModel:
        """Build JEPA ASR model.

        Args:
            args: Training arguments

        Returns:
            ESPnetJEPAASRModel instance
        """
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")

        # If use multi-blank transducer criterion,
        # big blank symbols are added just before the standard blank
        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:  # avoid this during inference
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        # 1. frontend
        if args.input_size is None or args.frontend in cls.feature_domain_frontends:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.input_size is not None and args.frontend in cls.feature_domain_frontends:
                args.frontend_conf = dict(args.frontend_conf)
                args.frontend_conf.setdefault("input_dim", args.input_size)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 5. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 6. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 7. Decoder
        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)

            if args.decoder == "transducer":
                from espnet2.asr_transducer.joint_network import JointNetwork

                decoder = decoder_class(
                    vocab_size,
                    embed_pad=0,
                    **args.decoder_conf,
                )

                joint_network = JointNetwork(
                    vocab_size,
                    encoder.output_size(),
                    decoder.dunits,
                    **args.joint_net_conf,
                )
            else:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                joint_network = None
        else:
            decoder = None
            joint_network = None

        # 8. CTC
        from espnet2.asr.ctc import CTC

        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 9. Build JEPA model
        try:
            model_class = jepa_model_choices.get_class(args.model)
        except AttributeError:
            model_class = ESPnetJEPAASRModel
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            token_list=token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 10. Initialize
        if args.init is not None:
            from espnet2.torch_utils.initialize import initialize

            initialize(model, args.init)

        return model
