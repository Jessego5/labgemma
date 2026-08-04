"""
Caption scoring and confidence tiering.

This is the quality layer the project actually turns on -- collection is cheap
now, so what matters is deciding which pairs are worth training on.

Scores produced per figure:

  keyword_score    0-1. Weighted density of gel/blot vocabulary in the caption.
  gel_score        0-1. Heuristic likelihood the figure IS a gel/blot image,
                   combining keyword evidence with penalties for terms that
                   suggest a chart or schematic.
  caption_quality  0-1. Is the caption long enough, specific enough, and free
                   of the markers of a caption that describes something else?
  compound_figure  bool. Does the caption enumerate multiple panels?
  tier             high / medium / low, per the project plan's definitions.

NOTE ON gel_score: this is a KEYWORD HEURISTIC, not a trained classifier. It is
deliberately not named `gel_probability` because it is not calibrated and does
not represent a probability. Swapping in a small image classifier later is the
natural upgrade; nothing downstream depends on how this number is produced.
"""

import re

from . import config

# Panel enumeration: "(A)", "(a)", "A.", "(A-C)", "(i)". Two or more distinct
# panel markers is strong evidence of a compound figure.
_PANEL_RE = re.compile(r"\(\s*([A-Za-z]|[ivx]{1,4})\s*[\)\-–]")
_PANEL_ALT_RE = re.compile(r"\b([A-H])\s*[\).]\s+[A-Z]")


def _norm(caption: str) -> str:
    return re.sub(r"\s+", " ", (caption or "")).strip()


def _hits(text: str, keywords) -> list:
    low = text.lower()
    return [k for k in keywords if k in low]


def keyword_score(caption: str) -> float:
    """Weighted density of assay vocabulary. High terms count triple."""
    high = _hits(caption, config.HIGH_KEYWORDS)
    med = _hits(caption, config.MEDIUM_KEYWORDS)
    raw = 3.0 * len(high) + 1.0 * len(med)
    # Saturating: 3 high-confidence hits is already maximal evidence.
    return min(1.0, raw / 9.0)


def compound_figure(caption: str) -> bool:
    """True if the caption enumerates two or more panels."""
    marks = {m.group(1).upper() for m in _PANEL_RE.finditer(caption)}
    marks |= {m.group(1).upper() for m in _PANEL_ALT_RE.finditer(caption)}
    return len(marks) >= 2


def caption_quality(caption: str) -> float:
    """
    0-1 quality score. Penalizes captions that are too short, too long, or
    dominated by vocabulary suggesting a non-gel figure.
    """
    c = _norm(caption)
    words = c.split()
    n = len(words)
    if n < config.MIN_CAPTION_WORDS:
        return 0.0

    score = 1.0
    # Very long captions are usually multi-panel omnibus captions where only
    # part of the text describes the blot.
    if n > config.MAX_CAPTION_WORDS:
        score -= 0.3
    elif n > 200:
        score -= 0.1
    # Short-but-valid captions carry less signal.
    if n < 20:
        score -= 0.2

    n_sus = len(_hits(c, config.SUSPICIOUS_KEYWORDS))
    score -= 0.15 * n_sus

    # A caption naming the assay explicitly is more trustworthy.
    if _hits(c, config.HIGH_KEYWORDS):
        score += 0.15

    return max(0.0, min(1.0, score))


def gel_score(caption: str) -> float:
    """
    Heuristic 0-1 that the figure is a gel/blot IMAGE (not a chart about one).
    """
    high = _hits(caption, config.HIGH_KEYWORDS)
    med = _hits(caption, config.MEDIUM_KEYWORDS)
    sus = _hits(caption, config.SUSPICIOUS_KEYWORDS)

    if high:
        base = 0.75 + 0.05 * min(len(high), 3)
    elif med:
        base = 0.35 + 0.05 * min(len(med), 4)
    else:
        base = 0.05

    base -= 0.12 * len(sus)
    # Compound figures mix a blot panel with charts, diluting the pairing.
    if compound_figure(caption):
        base -= 0.08
    return max(0.0, min(1.0, base))


def tier(caption: str) -> str:
    """
    Confidence tier per the project plan.

      high   - caption names the assay outright, is clean and specific
      medium - gel/blot vocabulary without naming the assay
      low    - ambiguous, too short, or looks like a chart/schematic
    """
    c = _norm(caption)
    if len(c.split()) < config.MIN_CAPTION_WORDS:
        return "low"

    high = _hits(c, config.HIGH_KEYWORDS)
    med = _hits(c, config.MEDIUM_KEYWORDS)
    q = caption_quality(c)
    g = gel_score(c)

    if high and q >= 0.6 and g >= 0.6:
        return "high"
    if high or (med and g >= 0.4):
        return "medium"
    if med:
        return "medium" if q >= 0.5 else "low"
    return "low"


def score_all(caption: str) -> dict:
    """All scores for one caption, as manifest columns."""
    c = _norm(caption)
    return {
        "keyword_score": round(keyword_score(c), 3),
        "gel_score": round(gel_score(c), 3),
        "caption_quality": round(caption_quality(c), 3),
        "compound_figure": compound_figure(c),
        "tier": tier(c),
    }


def is_candidate(caption: str) -> bool:
    """Cheap prefilter: keep anything with any gel/blot vocabulary at all."""
    c = _norm(caption).lower()
    if len(c.split()) < config.MIN_CAPTION_WORDS:
        return False
    return bool(_hits(c, config.HIGH_KEYWORDS) or _hits(c, config.MEDIUM_KEYWORDS))
