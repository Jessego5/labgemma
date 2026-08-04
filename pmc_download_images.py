"""
Step 2 (S3 version): download real figure images from the PMC Open Access S3
bucket and build the (image, caption) dataset for the Gemma harness.

Why S3: NCBI is retiring the old FTP .tar.gz packages (they 404 during the
2026 transition). The stable route is the public S3 bucket
  https://pmc-oa-opendata.s3.amazonaws.com/
where each article is a folder PMC{id}.{version}/ containing the figure images
directly (e.g. PMC13374667.1/AJMB-18-96-g002.jpg). No AWS account, no tarball,
no extraction -- just a direct HTTPS GET per image.

Per article:
  1. LIST the bucket for prefix "PMC{id}." to find the real version folder
     (usually .1, sometimes .2/.3 for revised articles).
  2. For each manifest row of that article, fetch
     https://pmc-oa-opendata.s3.amazonaws.com/{key} where key ends in image_href.
  3. Save the image + pair it with the caption.

Run:   python pmc_download_images.py
In:    pmc_gel_manifest.csv
Out:   data/datasets/gel-real/{images/*, train.csv, validation.csv}
Deps:  pip install requests
"""

import os
import csv
import re
import time
import xml.etree.ElementTree as ET
import requests

MANIFEST = "pmc_gel_manifest.csv"
OUT_DIR = "data/datasets/gel-real"
IMG_DIR = os.path.join(OUT_DIR, "images")
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
VALID_FRACTION = 0.15
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff")

# S3 ListBucketResult uses this XML namespace on every tag.
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_article_keys(pmcid):
    """
    List all object keys for an article by prefix 'PMC{id}.'.
    Returns list of keys, e.g. ['PMC123.1/fig1.jpg', 'PMC123.1/PMC123.1.xml'].
    Handles the version folder automatically (we don't assume .1).
    """
    url = f"{S3_BASE}/?list-type=2&prefix=PMC{pmcid}."
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    keys = [c.find(f"{S3_NS}Key").text
            for c in root.findall(f"{S3_NS}Contents")]
    return keys


def pick_image_key(keys, image_href):
    """Match the manifest image_href filename to an actual S3 key."""
    target = os.path.basename(image_href).lower()
    stem = os.path.splitext(target)[0]
    # exact filename match first
    for k in keys:
        if os.path.basename(k).lower() == target:
            return k
    # then stem match (handles extension differences)
    for k in keys:
        kb = os.path.basename(k).lower()
        if kb.endswith(IMAGE_EXTS) and os.path.splitext(kb)[0] == stem:
            return k
    # then loose contains-match
    for k in keys:
        kb = os.path.basename(k).lower()
        if kb.endswith(IMAGE_EXTS) and stem and stem in kb:
            return k
    return None


def download_key(key, dest_path):
    """Download one S3 object to dest_path."""
    url = f"{S3_BASE}/{key}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    by_article = {}
    for r in rows:
        by_article.setdefault(r["pmcid"], []).append(r)

    print(f"{len(rows)} figures across {len(by_article)} articles.\n")

    dataset_rows = []
    n_done = 0
    for i, (pmcid, figs) in enumerate(by_article.items(), 1):
        try:
            keys = list_article_keys(pmcid)
            if not keys:
                print(f"  [{i}/{len(by_article)}] PMC{pmcid}: not in S3 bucket, skip")
                continue
            got = 0
            for fig in figs:
                key = pick_image_key(keys, fig["image_href"])
                if not key:
                    print(f"      PMC{pmcid} {fig['fig_id']}: "
                          f"'{fig['image_href']}' not found among {len(keys)} keys")
                    continue
                ext = os.path.splitext(key)[1].lower()
                out_name = f"PMC{pmcid}_{fig['fig_id']}{ext}"
                download_key(key, os.path.join(IMG_DIR, out_name))
                caption = re.sub(r"\s+", " ", fig["caption"]).strip()
                dataset_rows.append({
                    "id": len(dataset_rows) + 1,
                    "image_path": f"images/{out_name}",
                    "caption": caption,
                    "pmcid": pmcid,
                    "license": fig.get("license", ""),
                })
                n_done += 1
                got += 1
                time.sleep(0.2)   # gentle between image fetches
            print(f"  [{i}/{len(by_article)}] PMC{pmcid}: {got}/{len(figs)} images")
        except Exception as e:
            print(f"  [{i}/{len(by_article)}] PMC{pmcid}: ERROR {e}")
        time.sleep(0.3)

    n = len(dataset_rows)
    n_val = max(1, int(n * VALID_FRACTION)) if n > 1 else 0
    val_rows, train_rows = dataset_rows[:n_val], dataset_rows[n_val:]

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "image_path", "caption", "pmcid", "license"])
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(OUT_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(OUT_DIR, "validation.csv"), val_rows)

    print(f"\nDONE: {n_done} images downloaded and paired with captions.")
    print(f"  train.csv:      {len(train_rows)} rows")
    print(f"  validation.csv: {len(val_rows)} rows")
    print(f"  images in:      {IMG_DIR}")
    print("\nZip data/datasets/gel-real/, move to the pod, add a [dataset:gel-real]")
    print("+ [profile:gel-real] block (copy gel-sample, rename), run finetune gel-real.")


if __name__ == "__main__":
    main()