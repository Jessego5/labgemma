"""
Central configuration for the LabGemma PMC pipeline.

EVERY large path is rooted at DATA_ROOT so the dataset lives on a RunPod
network volume, never on your local disk. Nothing bigger than a CSV is ever
written next to this source code.

Environment variables (all optional):

  LABGEMMA_DATA   where images/XML/manifests go.
                  Default: /workspace/labgemma if /workspace exists (RunPod),
                  else ./data next to the repo.
  NCBI_API_KEY    free key from https://account.ncbi.nlm.nih.gov/settings/
                  Raises the NCBI rate limit from 3 to 10 requests/sec.
  NCBI_EMAIL      contact address NCBI asks you to send with each request.

Check where your data will land before a big run:

    python -m pipeline.config
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def _default_root() -> Path:
    """Prefer an explicit override, then a RunPod volume, then the repo."""
    if env := os.environ.get("LABGEMMA_DATA"):
        return Path(env).expanduser()
    # RunPod network volumes are conventionally mounted at /workspace.
    if Path("/workspace").is_dir():
        return Path("/workspace/labgemma")
    return Path(__file__).resolve().parent.parent / "data"


DATA_ROOT = _default_root()

XML_CACHE = DATA_ROOT / "xml_cache"      # gzipped per-article JATS XML
IMAGE_DIR = DATA_ROOT / "images"         # downloaded figure images
MANIFEST = DATA_ROOT / "manifest.csv"    # one row per candidate figure
DOWNLOAD_LOG = DATA_ROOT / "download_log.csv"   # per-row download outcome
PAIRS = DATA_ROOT / "pairs.csv"          # final deduped + split dataset
STATS = DATA_ROOT / "stats.json"         # pipeline statistics

ALL_DIRS = [DATA_ROOT, XML_CACHE, IMAGE_DIR]

# --------------------------------------------------------------------------
# NCBI access
# --------------------------------------------------------------------------

EMAIL = os.environ.get("NCBI_EMAIL", "yyang784@wisc.edu")
API_KEY = os.environ.get("NCBI_API_KEY") or None

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"

# NCBI's published limits. We stay just under them.
RATE_LIMIT = 10.0 if API_KEY else 3.0    # requests/sec
MAX_RETRIES = 5                          # on 429/500/502/503/504
BACKOFF_BASE = 1.5                       # seconds; doubles each retry

# efetch accepts many IDs per call. Measured: 20 ids -> 2.9s (0.14s/article)
# versus 2.26s/article one at a time. This is the single biggest speed lever
# in the whole pipeline. Keep it well under NCBI's ~200-id practical ceiling.
EFETCH_BATCH = 50

# S3 is not rate limited by NCBI and is pure network latency, so we fan out.
S3_WORKERS = 16

# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

# esearch refuses retstart > 9998, so any query matching more than ~10k
# articles must be split into date slices. pipeline/ncbi.py does that
# automatically by bisecting the range below until each slice fits.
# Measured 2026-08-03 on a ~110-article sample per term:
#
#   term                 pool       high-tier/article   articles for 100k
#   NARROW (Title/Abs)   77,370     1.68                59,375  (77% of pool)
#   BROAD  (All Fields)  928,659    0.97                102,804 (11% of pool)
#
# NARROW is denser -- a paper with "western blot" in its title really is about
# blots. But hitting 100k pairs would consume 77% of that pool, leaving no
# headroom if the yield estimate is off. BROAD needs 1.7x the articles but
# uses 11% of its pool, so it tolerates a 2x yield miss and still scales.
#
# To combine: run NARROW first (best density), then re-run with BROAD and
# --resume. PMCIDs dedupe against the existing manifest and revisited
# articles hit the XML cache, so the second pass only pays for new ones.
SEARCH_TERM = (
    '("western blot"[All Fields] OR "immunoblot"[All Fields] '
    'OR "SDS-PAGE"[All Fields] OR "gel electrophoresis"[All Fields] '
    'OR "agarose gel"[All Fields]) AND open access[filter]'
)

NARROW_SEARCH_TERM = (
    "(western blot[Title/Abstract] OR gel electrophoresis[Title/Abstract]) "
    "AND open access[filter]"
)
DATE_FROM = "2000/01/01"
DATE_TO = "2026/12/31"
ESEARCH_PAGE_LIMIT = 9998   # NCBI hard cap on retstart

# When sampling fewer articles than the pool holds, split the date range into
# roughly this many slices and draw proportionally from each. Spreads the
# sample across the whole timeline instead of clustering in the oldest years.
SAMPLE_SLICES = 40

# --------------------------------------------------------------------------
# Caption filtering and confidence tiers
# --------------------------------------------------------------------------

# HIGH: the caption names the assay outright. Main train/eval set.
HIGH_KEYWORDS = [
    "western blot", "westernblot", "immunoblot", "immuno-blot",
    "sds-page", "sds page", "gel electrophoresis", "agarose gel",
    "polyacrylamide gel", "native page", "northern blot", "southern blot",
]

# MEDIUM: gel/blot vocabulary without naming the assay. Noisy extra data.
MEDIUM_KEYWORDS = [
    "blot", "electrophoresis", "agarose", "coomassie", "loading control",
    "lane", "kda", "molecular weight", "densitometry", "band intensity",
    "gapdh", "β-actin", "beta-actin", "b-actin", "α-tubulin",
    "protein expression", "membrane", "band",
]

# Terms suggesting the figure is a chart or schematic that merely mentions a
# blot, not an image of one. Pushes a caption down a tier.
SUSPICIOUS_KEYWORDS = [
    "schematic", "diagram", "flowchart", "flow chart", "workflow",
    "scatter plot", "survival curve", "kaplan", "flow cytometry",
    "immunofluorescence", "confocal", "h&e", "hematoxylin",
    "bar graph", "line graph", "heatmap", "volcano plot",
]

# Captions below this many words carry too little signal to train on.
MIN_CAPTION_WORDS = 8
MAX_CAPTION_WORDS = 400

# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------

# Paper-level split, so no article contributes figures to two splits.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 20260803

# Tiers admitted into the built dataset. Overridden by --tiers on the CLI.
DEFAULT_TIERS = ["high"]


def ensure_dirs() -> None:
    """Create the data tree. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable summary of the resolved configuration."""
    free_gb = None
    try:
        st = os.statvfs(DATA_ROOT if DATA_ROOT.exists() else DATA_ROOT.parent)
        free_gb = st.f_bavail * st.f_frsize / 1e9
    except OSError:
        pass
    lines = [
        f"DATA_ROOT   {DATA_ROOT}",
        f"  exists    {DATA_ROOT.exists()}",
        f"  free disk {free_gb:.1f} GB" if free_gb is not None else "  free disk unknown",
        f"NCBI key    {'set (10 req/s)' if API_KEY else 'NOT set (3 req/s)'}",
        f"NCBI email  {EMAIL}",
        f"search      {SEARCH_TERM}",
        f"dates       {DATE_FROM} .. {DATE_TO}",
        f"batching    {EFETCH_BATCH} ids/efetch, {S3_WORKERS} S3 workers",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
