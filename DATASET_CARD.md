# LabGemma-PMC-Gel

A licensed dataset of biomedical **gel and blot** figure–caption pairs mined
from the PubMed Central Open Access subset, with confidence tiering, license
classification, and provenance for every row.

Released as **metadata plus image references**, not image bytes. Each row
carries the article's PMCID, its versioned S3 object key, and its license, so
images are fetched directly from NCBI's public bucket by the included
downloader. This keeps redistribution within what each article's license
allows and keeps the release small.

---

## At a glance

| | |
|---|---|
| Rows (full manifest) | 202,224 figures |
| Articles | 73,691 |
| Domain | Western blots, immunoblots, SDS-PAGE, agarose gels |
| Source | PubMed Central Open Access subset |
| Public release | rows with `license_class` ∈ {CC BY, CC0} |
| Language | English |
| Collection date | August 2026 |

## Why gel/blot only

Existing PMC-derived datasets — PMC-OA (1.6M), PMC-15M, Open-PMC-18M — span
microscopy, radiography, histology and charts at once. Restricting to a single
assay type makes the **text channel well defined**: in a western blot, protein
symbols and molecular weights are printed beside the bands, and captions name
those same proteins. That makes questions like *how much of figure–caption
alignment is reading printed labels?* answerable, which they are not on a
heterogeneous corpus.

---

## Schema

| field | description |
|---|---|
| `pmcid` | PubMed Central ID of the source article |
| `figure_id` | JATS `<fig>` element id |
| `label` | figure label as printed, e.g. "Figure 3" |
| `caption` | full caption text, whitespace-normalised |
| `image_href` | filename referenced by the article's `<graphic>` |
| `s3_prefix` | versioned OA bucket folder, e.g. `PMC3219872.2` |
| `license_url` | machine-readable license from NISO `ali:license_ref` |
| `license` | license type attribute and prose |
| `license_class` | `CC BY`, `CC BY-NC`, `CC0`, `PUBLISHER-FREE`, … |
| `keyword_score` | 0–1, weighted assay-vocabulary density |
| `gel_score` | 0–1 heuristic that the figure is a gel/blot image |
| `caption_quality` | 0–1, length and specificity |
| `compound_figure` | caption enumerates ≥2 panels |
| `tier` | `high` / `medium` / `low` confidence |
| `n_figs_in_article` | figure count, for same-paper negative sampling |

The full image is at `https://pmc-oa-opendata.s3.amazonaws.com/{s3_prefix}/{image_href}`.

## Composition

**Confidence tiers** (keyword-derived, not model-derived):

| tier | rows | definition |
|---|---|---|
| `high` | 133,236 | caption names the assay outright (western blot, immunoblot, SDS-PAGE, agarose gel) |
| `medium` | 67,999 | gel/blot vocabulary without naming the assay (lanes, kDa, loading control, GAPDH) |
| `low` | 989 | too short, or dominated by chart/schematic vocabulary |

**Licenses** across the full manifest:

| class | rows | redistributable |
|---|---|---|
| CC BY | 83,092 | yes, with attribution |
| CC BY-NC-ND | 52,385 | no derivatives |
| CC BY-NC | 23,526 | non-commercial |
| CC BY-NC-SA | 15,874 | non-commercial, share-alike |
| OTHER | 16,999 | case by case |
| UNKNOWN | 4,163 | no |
| PUBLISHER-FREE | 4,014 | **no** — free to read, no reuse grant |
| PMC-OA-ONLY | 1,099 | PMC-internal terms |

`PUBLISHER-FREE` is mostly Elsevier's COVID-era free-access notice. Those
articles are readable but grant **no** reuse rights, and are labelled rather
than silently mixed in — an early version of this pipeline classified 90% of
rows as "OTHER" by reading license prose instead of the `ali:license_ref` URL.

**72.1%** of figures are compound (multi-panel).

---

## Derived benchmark

The repository also builds a figure–caption **matching** benchmark:

- **positives** — the published figure–caption pair
- **easy negatives** — a caption from a different article
- **hard negatives** — a different figure's caption from the **same article**

Same-paper hard negatives are the biomedical analogue of the complementary
image pairs in Goyal et al., *Making the V in VQA Matter* (CVPR 2017): both
captions come from one paper, so "which paper is this?" carries no signal and
paper-level memorisation from pretraining is neutralised. 79% of figures have a
same-paper sibling available.

Splits are assigned by **hashing the PMCID**, so no article contributes figures
to two splits and an article's split never changes as the dataset grows.

---

## Limitations

**Positives are weak labels.** "This caption describes this figure" comes from
co-publication; nobody verified it. With 72% compound figures, most captions
describe several panels of which only one is a blot.

**Negatives can be semantically true.** "GAPDH was used as a loading control"
is literally true of thousands of blots here, so a negative may be accurate
while labelled 0. This concentrates in the `medium` tier.

**Captions describe context, not appearance.** Measured on this corpus, only
22% of caption sentences contain anything visually checkable; 87% carry
statistics and 49% describe sample provenance. This is a property of biomedical
writing, not a defect in extraction — but it bounds what caption supervision
can teach.

**Tiers and `gel_score` are heuristics.** Keyword-derived, uncalibrated. The
field is deliberately not named `gel_probability`.

**No manual verification.** No human has checked that any caption matches its
figure.

**Coverage is a sample.** 103,000 of 928,659 matching articles, drawn
proportionally across date slices from 2000–2026.

---

## Licensing

Metadata rows are facts about published articles. **Caption text is quoted from
the source articles** and carries each article's own license, recorded per row.

The public release includes only `CC BY` and `CC0` rows. To reproduce other
subsets, run the pipeline with `--licenses`.

Attribution for any CC BY row is its `pmcid` plus `license_url`.

Images are **not** redistributed; they are fetched from NCBI's public S3 bucket
by `pipeline/download.py`.

## Reproduction

```bash
export LABGEMMA_DATA=/path/to/data
python -m pipeline.extract  --max-articles 103000
python -m pipeline.download --tiers high --licenses "CC BY" CC0
python -m pipeline.dataset  --tiers high --licenses "CC BY" CC0
python -m pipeline.task
```

Collection takes ~19 minutes for 103,000 articles. Every stage is resumable.
Search terms, date range, tier keywords and split seed are in
`pipeline/config.py`.

## Citation

```
@misc{labgemma_pmc_gel_2026,
  title  = {LabGemma-PMC-Gel: licensed biomedical gel/blot figure-caption pairs
            from PubMed Central},
  author = {Yang, Jessica},
  year   = {2026},
  url    = {https://github.com/Jessego5/labgemma}
}
```

Source articles remain © their authors under the licenses recorded per row.
PMC data is provided by the U.S. National Library of Medicine, which does not
endorse this derivative.
