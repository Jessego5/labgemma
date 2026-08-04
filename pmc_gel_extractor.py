"""
PMC gel/blot figure-caption extractor (step 1 of building the real dataset).

What it does:
  1. SEARCH PubMed Central's open-access subset for articles about gels/blots
     (esearch), restricted to the OA subset so we can legally use the figures.
  2. FETCH each article's full-text XML (efetch).
  3. PARSE the XML for <fig> elements -> pull the caption text, figure label,
     and the image file reference (the graphic's href).
  4. FILTER captions to ones actually about gels/blots/plates (keyword match).
  5. RECORD provenance: PMCID + license + figure id + image href, so every pair
     is auditable and you can fetch the actual image files in step 2.

This step produces a MANIFEST (CSV) of candidate figures. It does NOT download
the images yet -- that's the next step, and it lets you eyeball/filter the
manifest first (quality control before you pull image bytes).

Run:   python pmc_gel_extractor.py
Out:   pmc_gel_manifest.csv
Deps:  pip install requests   (uses stdlib xml, no pubmed_parser needed)

NOTE: set EMAIL below. NCBI asks that E-utilities requests identify you, and it
raises your rate limit. Be polite: <=3 requests/sec without an API key.
"""

import requests
import xml.etree.ElementTree as ET
import csv
import time
import re

EMAIL = "yyang784@wisc.edu"          # <-- EDIT to your email (NCBI politeness)
NCBI_API_KEY = None                # optional: paste an NCBI API key to go faster

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# How many articles to pull in this run. Start SMALL (25-50) to sanity-check,
# then raise once you trust the output.
MAX_ARTICLES = 30

#CAN CHANGE LATER!!!!!!!!!!!!
# Search query. "open access"[filter] restricts to the OA subset (reusable).
# We bias toward western blots / gels specifically.
SEARCH_TERM = '(western blot[Title/Abstract] OR gel electrophoresis[Title/Abstract]) AND open access[filter]'

# Caption keywords that mark a figure as a lab readout we care about.
LAB_KEYWORDS = [
    "western blot", "immunoblot", "blot", "sds-page", "sds page",
    "gel electrophoresis", "agarose", "electrophoresis", "lane", "kda",
    "coomassie", "loading control",
]


def esearch(term, retmax):
    """Return a list of PMCIDs matching the search term in the OA subset."""
    params = {
        "db": "pmc",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "email": EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    idlist = r.json().get("esearchresult", {}).get("idlist", [])
    return idlist


def efetch_xml(pmcid):
    """Fetch one article's full-text XML by PMCID (numeric, no 'PMC' prefix)."""
    params = {"db": "pmc", "id": pmcid, "retmode": "xml", "email": EMAIL}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.text


def text_of(elem):
    """Flatten an XML element's text content (captions have nested tags)."""
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def parse_figures(xml_text, pmcid):
    """
    Pull figures from article XML. Returns list of dicts:
      {pmcid, license, fig_id, label, caption, image_href}
    Defensive against JATS structural variation.
    """
    figs = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return figs

    # License: JATS puts it under article-meta/permissions/license.
    # We grab license-type attr and/or the license text if present.
    license_info = ""
    for lic in root.iter("license"):
        lt = lic.get("license-type") or lic.get("{http://www.w3.org/1999/xlink}href") or ""
        license_info = (lt + " " + text_of(lic)).strip()[:120]
        if license_info:
            break

    for fig in root.iter("fig"):
        fig_id = fig.get("id", "")
        label = text_of(fig.find("label"))
        caption = text_of(fig.find("caption"))
        # image reference: <graphic xlink:href="...">
        href = ""
        for g in fig.iter("graphic"):
            href = (g.get("{http://www.w3.org/1999/xlink}href")
                    or g.get("href") or "")
            if href:
                break
        figs.append({
            "pmcid": pmcid,
            "license": license_info,
            "fig_id": fig_id,
            "label": label,
            "caption": caption,
            "image_href": href,
        })
    return figs


def is_lab_readout(caption):
    """True if the caption looks like a gel/blot/plate figure."""
    c = caption.lower()
    return any(k in c for k in LAB_KEYWORDS)


def main():
    if "example.com" in EMAIL:
        print("!! Edit EMAIL at the top of this file to your own address first.\n")

    print(f"Searching PMC OA for: {SEARCH_TERM}")
    pmcids = esearch(SEARCH_TERM, MAX_ARTICLES)
    print(f"Found {len(pmcids)} articles (showing up to {MAX_ARTICLES}).\n")

    all_rows = []
    kept = 0
    for i, pmcid in enumerate(pmcids, 1):
        try:
            xml = efetch_xml(pmcid)
            figs = parse_figures(xml, pmcid)
            lab_figs = [f for f in figs if is_lab_readout(f["caption"])]
            all_rows.extend(lab_figs)
            kept += len(lab_figs)
            print(f"  [{i}/{len(pmcids)}] PMC{pmcid}: "
                  f"{len(figs)} figures, {len(lab_figs)} gel/blot-relevant")
        except Exception as e:
            print(f"  [{i}/{len(pmcids)}] PMC{pmcid}: ERROR {e}")
        time.sleep(0.34)  # ~3 req/sec, polite without an API key

    # Write the manifest
    out = "pmc_gel_manifest.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pmcid", "license", "fig_id", "label", "image_href", "caption"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nWrote {kept} candidate gel/blot figures to {out}")
    print("\nNEXT STEPS:")
    print("  1. Open the manifest and EYEBALL it -- are the captions really")
    print("     describing gels/blots? Delete rows that aren't.")
    print("  2. Check the 'license' column -- keep CC BY / CC0 for safe reuse.")
    print("  3. Then (step 2 script) download the image_href files per article")
    print("     from the PMC OA package, pairing each image with its caption.")
    print("\nStart with MAX_ARTICLES small; raise it once the output looks clean.")


if __name__ == "__main__":
    main()