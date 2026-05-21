"""ASR Task for SE-JEPA: Conv-TasNet (SI-SNR) + JEPA predictive + Conformer CTC."""

import argparse

from typeguard import typechecked

from espnet2.asr.espnet_se_jepa_model import ESPnetSEJEPAASRModel
from espnet2.asr.frontend.se_jepa import SE_JEPAFrontend
from espnet2.tasks.asr_jepa import (
    ASRJEPATask,
    decoder_choices,
    encoder_choices,
    frontend_choices,
    jepa_model_choices,
    normalize_choices,
    postencoder_choices,
    preencoder_choices,
    preprocessor_choices,
    specaug_choices,
)
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices

# Extend frontend with SE-JEPA
se_jepa_frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        **frontend_choices.classes,
        se_jepa=SE_JEPAFrontend,
    ),
    type_check=frontend_choices.base_type,
    default=frontend_choices.default,
)

# Extend model with SE-JEPA
se_jepa_model_choices = ClassChoices(
    "model",
    classes=dict(
        **jepa_model_choices.classes,
        se_jepa_espnet=ESPnetSEJEPAASRModel,
    ),
    type_check=AbsESPnetModel,
    default="se_jepa_espnet",
)


class ASRSEJEPATask(ASRJEPATask):
    """ASR Task for SE-JEPA: SE(wav) → log-mel → JEPA predictive → ASR."""

    class_choices_list = [
        se_jepa_frontend_choices,
        specaug_choices,
        normalize_choices,
        se_jepa_model_choices,
        preencoder_choices,
        encoder_choices,
        postencoder_choices,
        decoder_choices,
        preprocessor_choices,
    ]

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSEJEPAASRModel:
        """Build SE-JEPA ASR model. Same as ASRJEPATask but uses se_jepa frontend/model choices."""
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")

        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)

        if args.input_size is None:
            frontend_class = se_jepa_frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        if args.specaug is not None:
            specaug = specaug_choices.get_class(args.specaug)(**args.specaug_conf)
        else:
            specaug = None

        if args.normalize is not None:
            normalize = normalize_choices.get_class(args.normalize)(**args.normalize_conf)
        else:
            normalize = None

        if getattr(args, "preencoder", None) is not None:
            preencoder = preencoder_choices.get_class(args.preencoder)(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        encoder = encoder_choices.get_class(args.encoder)(input_size=input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size()

        if getattr(args, "postencoder", None) is not None:
            postencoder = postencoder_choices.get_class(args.postencoder)(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)
            if args.decoder == "transducer":
                from espnet2.asr_transducer.joint_network import JointNetwork
                decoder = decoder_class(vocab_size, embed_pad=0, **args.decoder_conf)
                joint_network = JointNetwork(vocab_size, encoder.output_size(), decoder.dunits, **args.joint_net_conf)
            else:
                decoder = decoder_class(vocab_size=vocab_size, encoder_output_size=encoder_output_size, **args.decoder_conf)
                joint_network = None
        else:
            decoder = None
            joint_network = None

        from espnet2.asr.ctc import CTC
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf)

        model_class = se_jepa_model_choices.get_class(args.model)
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

        if args.init is not None:
            from espnet2.torch_utils.initialize import initialize
            initialize(model, args.init)

        return model
