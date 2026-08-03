"""Small shared value types for Enhancer-compatible nodes."""

from dataclasses import dataclass


REFERENCE_LATENTS_KEY = "reference_latents"
REFERENCE_METHOD_KEY = "reference_latents_method"
KLEIN_SECTIONS_KEY = "klein_sections"

ENHANCER_CATEGORY = "Nunchaku/FLUX.2 Klein/Enhancer"
REFERENCE_CATEGORY = f"{ENHANCER_CATEGORY}/Reference"
TEXT_CATEGORY = f"{ENHANCER_CATEGORY}/Text"
POST_CFG_CATEGORY = f"{ENHANCER_CATEGORY}/Post-CFG"


@dataclass(frozen=True)
class TokenRange:
    """Half-open token range used by conditioning section metadata."""

    start: int
    end: int


@dataclass(frozen=True)
class KleinSections:
    """Validated immutable form of ``metadata["klein_sections"]``."""

    front: TokenRange
    mid: TokenRange
    end: TokenRange


__all__ = [
    "ENHANCER_CATEGORY",
    "KLEIN_SECTIONS_KEY",
    "KleinSections",
    "POST_CFG_CATEGORY",
    "REFERENCE_CATEGORY",
    "REFERENCE_LATENTS_KEY",
    "REFERENCE_METHOD_KEY",
    "TEXT_CATEGORY",
    "TokenRange",
]
