"""Initialize espnet2 package."""

import warnings

# Suppress librosa's pkg_resources deprecation warning (third-party; cannot fix upstream)
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=UserWarning,
)

from espnet import __version__  # NOQA
