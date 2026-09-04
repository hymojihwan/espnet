"""Frozen speech enhancement with Meta-BRIDGE adaptation."""

from typing import Any, Dict, Optional

from espnet2.asr.frontend.meta_bridge import MetaBridgeFrontend
from espnet2.asr.frontend.se_jepa import SE_JEPAFrontend


class SEMetaBridgeFrontend(SE_JEPAFrontend):
    """Compose a frozen enhancement model and Meta-BRIDGE frontend."""

    def __init__(
        self,
        meta_bridge_conf: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if "jepa_frontend" in kwargs or "jepa_frontend_conf" in kwargs:
            raise ValueError(
                "SEMetaBridgeFrontend accepts meta_bridge_conf instead of "
                "jepa_frontend or jepa_frontend_conf"
            )
        meta_bridge = MetaBridgeFrontend(**(meta_bridge_conf or {}))
        super().__init__(jepa_frontend=meta_bridge, **kwargs)
