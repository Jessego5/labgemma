"""
OCR utilities: read the text rendered inside figures, and mask it out.

Why this exists. Measured on real gel/blot captions: 100% contain protein or
gene symbols (median 5 per caption), and two figure captions from the SAME
paper share a median symbol Jaccard of only 0.31 -- so symbols both appear in
captions and DISCRIMINATE between figures within a paper. In a western blot,
those symbols are printed next to the bands.

That makes "read the labels in the image, match them to the caption" a viable
strategy that requires no learning and no understanding of the biology. This
module exists to measure exactly how far that strategy gets, two ways:

  extract  -> pipeline.baselines scores caption-vs-image symbol overlap
  mask     -> blank the text regions, so a trained model's drop on masked
              images is the share of its performance that was reading text

OCR is slow (~0.3-1s per image), so results are cached to DATA_ROOT/ocr.json
keyed by image filename. Re-runs are free.

Run:
    python -m pipeline.ocr extract          # OCR every image in pairs.csv
    python -m pipeline.ocr mask --split test  # write text-masked variants
"""

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config

# Same symbol heuristic used to characterise the captions: uppercase
# alphanumeric tokens, minus lab boilerplate that carries no identity.
STOPWORDS = {
    "DNA", "RNA", "PCR", "SDS", "PAGE", "WB", "IHC", "ELISA", "FBS", "PBS",
    "KDA", "MW", "BSA", "EDTA", "TBS", "HRP", "IP", "KO", "WT", "NC", "SI",
    "AND", "THE", "OF", "IN", "FOR", "WITH", "ANOVA", "SEM", "SD", "CI",
    "OR", "AS", "AT", "TO", "BY", "ON", "NS", "CTRL", "CON", "UN",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "II", "III", "IV", "V", "VI",
}
SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]{1,4})?)\b")

# tesseract confidence below this is usually noise from band edges.
MIN_CONF = 40


def symbols(text: str) -> set:
    """Protein/gene-like symbols in a string."""
    return {s for s in SYMBOL_RE.findall(text or "")
            if s not in STOPWORDS and not s.isdigit()}


def _tesseract():
    try:
        import pytesseract  # noqa
        from PIL import Image  # noqa
        return pytesseract, Image
    except ImportError as e:
        raise SystemExit(
            "OCR needs tesseract + pytesseract + pillow:\n"
            "  apt-get update && apt-get install -y tesseract-ocr\n"
            "  pip install pytesseract pillow numpy\n"
            f"({e})")


def ocr_image(path):
    """
    Return {"text": str, "boxes": [[x, y, w, h], ...]} for one image.

    Boxes are only kept for tokens above MIN_CONF, since low-confidence
    detections on blot images are usually band edges rather than text, and
    masking those would destroy image content rather than text.
    """
    pytesseract, Image = _tesseract()
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return {"text": "", "boxes": []}
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return {"text": "", "boxes": []}

    words, boxes = [], []
    for i, w in enumerate(d.get("text", [])):
        w = (w or "").strip()
        if not w:
            continue
        try:
            conf = float(d["conf"][i])
        except (ValueError, KeyError, IndexError):
            conf = -1.0
        if conf < MIN_CONF:
            continue
        words.append(w)
        boxes.append([int(d["left"][i]), int(d["top"][i]),
                      int(d["width"][i]), int(d["height"][i])])
    return {"text": " ".join(words), "boxes": boxes}


def cache_path(masked=False):
    """
    Separate caches for original and masked images.

    Masked variants keep the SAME filename in a different directory, so a
    single cache keyed by basename would silently overwrite the original
    OCR with the masked one -- and the ablation would then compare a thing
    against itself.
    """
    return config.DATA_ROOT / ("ocr_masked.json" if masked else "ocr.json")


def load_cache(masked=False) -> dict:
    p = cache_path(masked)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict, masked=False) -> None:
    p = cache_path(masked)
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(cache))
    tmp.replace(p)


def image_paths_from_pairs(split=None):
    """Every image referenced by pairs.csv, optionally one split only."""
    with open(config.PAIRS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    seen, out = set(), []
    for r in rows:
        p = config.DATA_ROOT / r["image_path"]
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


def cmd_extract(args) -> int:
    paths = image_paths_from_pairs(args.split)
    if args.masked:
        # Same filenames, different directory.
        mdir = config.DATA_ROOT / "images_masked"
        paths = [mdir / p.name for p in paths if (mdir / p.name).exists()]
        if not paths:
            print(f"No masked images in {mdir}. "
                  f"Run: python -m pipeline.ocr mask")
            return 2

    cache = load_cache(args.masked)
    todo = [p for p in paths if p.name not in cache]
    print(f"{len(paths):,} images{' (masked)' if args.masked else ''}, "
          f"{len(cache):,} cached, {len(todo):,} to OCR")
    if not todo:
        print("nothing to do")
        return 0

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ocr_image, p): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                cache[p.name] = fut.result()
            except Exception as e:
                cache[p.name] = {"text": "", "boxes": [], "error": str(e)}
            done += 1
            if done % 250 == 0:
                save_cache(cache, args.masked)
                print(f"  [{done:,}/{len(todo):,}]")
    save_cache(cache, args.masked)

    n_text = sum(1 for v in cache.values() if v.get("text"))
    n_sym = sum(1 for v in cache.values() if symbols(v.get("text", "")))
    print(f"\nOCR done: {len(cache):,} images cached")
    print(f"  with any text     {n_text:,} ({n_text/max(len(cache),1):.0%})")
    print(f"  with >=1 symbol   {n_sym:,} ({n_sym/max(len(cache),1):.0%})")
    print(f"  cache             {cache_path(args.masked)}")
    return 0


# --------------------------------------------------------------------------
# mask
# --------------------------------------------------------------------------


def cmd_mask(args) -> int:
    """
    Write text-masked copies of the images.

    Text boxes are filled with the median colour of the surrounding image
    rather than blurred, so the text is definitively gone -- a blur can leave
    enough structure for a model to still read it, which would understate the
    ablation and make the result look better than it is.
    """
    _, Image = _tesseract()
    import numpy as np
    from PIL import ImageDraw

    cache = load_cache()
    if not cache:
        print("No OCR cache. Run: python -m pipeline.ocr extract")
        return 2

    out_dir = config.DATA_ROOT / "images_masked"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths_from_pairs(args.split)

    n_done = n_boxes = n_skip = 0
    for p in paths:
        dest = out_dir / p.name
        if dest.exists() and not args.force:
            n_skip += 1
            continue
        rec = cache.get(p.name)
        if rec is None:
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(img)
        fill = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))
        draw = ImageDraw.Draw(img)
        for (x, y, w, h) in rec.get("boxes", []):
            pad = 2
            draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=fill)
            n_boxes += 1
        img.save(dest)
        n_done += 1
        if n_done % 250 == 0:
            print(f"  [{n_done:,}/{len(paths):,}]")

    print(f"\nmasked {n_done:,} images ({n_skip:,} already present)")
    print(f"  {n_boxes:,} text boxes filled "
          f"({n_boxes/max(n_done,1):.1f} per image)")
    print(f"  -> {out_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="OCR images and cache the result")
    e.add_argument("--split", default=None, help="test / val / train")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--masked", action="store_true",
                   help="OCR the masked variants instead (validates masking)")
    e.set_defaults(func=cmd_extract)

    m = sub.add_parser("mask", help="write text-masked image variants")
    m.add_argument("--split", default="test")
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_mask)

    args = ap.parse_args(argv)
    config.ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
