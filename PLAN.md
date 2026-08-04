# LabGemma — One-Week Plan

**Window:** Aug 3–10, 2026 · **Application deadline:** ~Sept 24, 2026
**Rule for this week:** ship the core completely. Do not start anything on the
expansion ladder until every day below is done and committed.

---

## The question

> When a model matches a gel/blot figure to its caption, *what is it actually
> using* — text rendered inside the figure, panel layout, or visual content?
> And does hard-negative training change that?

Not "can we fine-tune Gemma on figure-caption pairs." That's been done at
1.6M–18M scale (MICCAI 2023, NeurIPS 2023, CVPR). What hasn't been done is
isolating the *mechanism* on a single homogeneous assay type, where rendered
text is well-defined and visual content is uniform. That's the contribution,
and it's achievable at small scale because it depends on controls, not volume.

---

## Scale at every stage

Every number below is measured, not estimated. Sources in parentheses.

| # | Stage | Scale | Disk | Time |
|---|---|---|---|---|
| 0 | Search pool (BROAD term) | **928,659 articles** | — | — |
| 1 | Articles fetched | **103,000** | 3.0 GB XML cache | ~3 hr |
| 2 | Manifest rows (candidates) | **~169,000** @ 1.64/article | 150 MB CSV | — |
| 3 | High-tier rows | **~100,000** @ 0.97/article | — | — |
| 4 | Images downloaded (subset) | **10,000** high-tier, CC-licensed | 1.3 GB | ~20 min |
| 5 | Pairs after dedup + split | **~9,800** across ~4,300 articles | — | seconds |
| 6 | → train / val / test (by paper) | **6,860 / 1,470 / 1,470** | — | — |
| 7 | Training rows per arm | **13,720** (1 negative per positive) | — | — |
| 8 | Frozen test rows | **~4,150** (1 pos + 1 easy + ~0.8 hard) | — | — |
| 9 | Text-masked test images | **1,470** | 0.2 GB | ~30 min |
| | **Total volume needed** | | **~20 GB** | |

**Why 100k manifest but only 10k images.** The manifest is the expensive
artifact — 3 hours of NCBI work — and costs 150 MB. Images are re-derivable
from it in minutes for any subset. This buys the "100k licensed dataset"
claim for 150 MB, while training only on what a 4090 can iterate on in a week.

**Why 0.97 high-tier/article.** Measured on the BROAD term. The NARROW term
yields 1.68/article but consuming 100k pairs would eat 77% of its pool; BROAD
uses 11%, so a 2× yield miss is survivable. See `pipeline/config.py`.

**Why ~80% hard-negative coverage.** Hard negatives require an article with
≥2 gel figures. Measured: 17/24 articles (71%) qualify, and because
figure-rich articles contribute more figures, ~80% of *figures* get one.

---

## Day by day

### Day 1 — Dataset + harness smoke test

Two tracks in parallel. The extract runs unattended; spend that time on the
harness, which is the only thing that can kill the week.

**Track A (unattended, ~3.5 hr)** — on the RunPod volume, not local disk:

```bash
export LABGEMMA_DATA=/workspace/labgemma
export NCBI_API_KEY=...            # free, raises 3 -> 10 req/s
python -m pipeline.config          # confirm where data lands
python -m pipeline.extract --max-articles 103000
python -m pipeline.download --tiers high --licenses "CC BY" "CC BY-NC" CC0 --limit 10000
python -m pipeline.dataset --tiers high --licenses "CC BY" "CC BY-NC" CC0
```

**Scale checkpoint:** manifest ≥90,000 high-tier rows, ≥9,500 images on disk,
download success rate ≥97%. If success rate is below 95%, re-run `download`
(it resumes) before moving on.

**Track B (active)** — build a 5-row CSV with 5 images and confirm one LoRA
step runs end to end in `gemma-tuner-multimodal`. Not a real train. Just proof
the path works.

> **If Track B is not working by end of day 1, abandon the harness and switch
> to a plain `transformers` + `peft` script.** Do not spend day 2 debugging
> someone else's wizard. Gemma multimodal LoRA is well-trodden in raw PEFT.

### Day 2 — Benchmark construction

Build `matching.csv` (binary) and `gallery.csv` (retrieval) from `pairs.csv`.

Task format is **VQA-as-matching**: *"Does this caption describe this figure?
Answer yes or no."* Chosen because it runs on the existing harness unmodified;
a contrastive dual-encoder would fit Recall@K better but needs a trainer you
don't have.

Per positive, emit exactly **one** negative:
- **A1 arm:** easy negative (different paper, same split)
- **A2 arm:** hard negative (different figure, *same paper*); easy if the
  article has only one figure

> **Both arms must have identical row counts and identical positives.** If A2
> gets extra rows because hard negatives are additive, you are measuring
> "more data" and not "better negatives," and the comparison is worthless.

Test set gets **both** negative types per positive so easy-vs-hard can be
reported as a breakdown.

**Hard invariants — set these today, never change them:**
- Negatives drawn **within-split only**. Cross-split negatives leak test
  captions into training and inflate every number.
- `SPLIT_SEED = 20260803` and `SPLIT_FRACTIONS` are **frozen**. The split is
  hashed from PMCID, so it stays stable as the dataset grows — that is what
  makes later expansion comparable to this week's results.
- Val/test negatives written to disk once and treated as **immutable**. Want a
  harder benchmark later? Add a second eval set. Never mutate this one.

**Scale checkpoint:** 13,720 train rows per arm, ~4,150 frozen test rows,
0 articles appearing in more than one split (`dataset.py` asserts this).

### Day 3 — Baselines + masked variants

**Run baselines before training anything.** If OCR-only already scores 0.85,
that reframes every decision that follows — and finding that out after a week
of training wastes the week.

1. **Random** — 50% on balanced binary. Free.
2. **OCR + symbol overlap** — tesseract on the image → uppercase symbols →
   Jaccard against caption symbols → threshold/rank. **No learning at all.**
   This is the single most important number in the project.
3. **Text-masked test images** — OCR-detect text boxes, inpaint, save 1,470
   masked variants for day 6.

Measured motivation: 100% of captions contain protein/gene symbols (median 5),
and same-paper caption pairs share a median Jaccard of only 0.31 — so symbols
*are* the discriminative signal, and they are printed inside blot images.

**Scale checkpoint:** OCR baseline scored on all ~4,150 test rows; 1,470
masked images written.

### Day 4 — Train A1 (easy negatives only)

13,720 rows, 2 epochs. Frozen vision encoder, LoRA on LM attention layers,
train the projector, BF16 on the 4090.

**Scale checkpoint:** ~1.5–2 hr. If a single arm exceeds 4 hours, halve the
dataset and retrain both arms at the smaller size rather than running arms of
different sizes.

### Day 5 — Train A2 (hard negatives)

Identical hyperparameters, identical positives, identical row count. **Only
the negative composition differs.** Any other change invalidates the
comparison.

### Day 6 — Evaluation + error analysis

Score both arms on the frozen test set:

- accuracy / precision / recall / F1 / AUROC
- **broken out by negative type** (easy vs hard) — the shortcut evidence
- **text-masked vs unmasked** — how much was reading labels
- Recall@1 / @5 on the gallery, if time remains

Error analysis on: compound figures (62.6% of the corpus), visually similar
blots from the same paper, captions dominated by non-visual context,
extraction errors.

### Day 7 — Ship

README (leads with the **finding**, not the architecture), model card,
data-license notes, architecture diagram, small inference demo.

Budget this day fully. An unwritten project is an unfinished one — a recruiter
spends ~90 seconds on the repo and almost never opens the code.

---

## What you'll be able to state

| Result | Reads as |
|---|---|
| OCR-only = **X** | How much needs no model at all |
| A1 (easy negatives) = **Y** | Standard recipe |
| A2 (hard negatives) = **Z** | Effect of hard-negative training |
| Both drop to **W** when text is masked | Share attributable to rendered text |
| Easy **A** vs hard **B** gap | Shortcut reliance, quantified |

Five numbers plus a benchmark artifact. Together they answer every objection
that can be raised against the project — including "isn't it just name
matching?" and "hasn't it already read the paper?"

---

## Cut order if a day slips

Drop from the bottom:

1. Recall@K gallery (keep binary metrics)
2. Text-masked ablation
3. Arm A1 — compare A2 against OCR + zero-shot base Gemma instead

**Never cut:** the OCR baseline, the easy-vs-hard breakdown, or day 7.

A project with the OCR baseline and no fine-tuning is defensible. A project
with two LoRA arms and no baselines is not interpretable.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Harness doesn't train images** | Smoke-test day 1. Fall back to raw `transformers` + `peft` immediately |
| Arms end up different sizes | Fix at 1 negative per positive. Verify row counts match before training |
| Negative sampler leaks across splits | Assert `negative.split == positive.split` in the sampler; `dataset.py` already checks article-level leakage |
| Download truncates silently | `download.py` exits non-zero and prints success rate. Re-run to resume |
| Local disk fills | Everything routes through `LABGEMMA_DATA` on the pod volume. Never run stage 1 locally |

---

## After this week (do not start early)

Expansion ladder, cheapest first. Items 1–3 complete the original four-arm
design and are roughly a weekend:

1. **Arm A3** — `--tiers high medium` (the weak-label-quality half of the
   original question)
2. **Arm A4** — `--drop-compound` (tests the panel-layout shortcut)
3. **Pre/post-cutoff** contamination test (one `--date-from`)
4. Scale to 30–50k images
5. BiomedCLIP baseline
6. Recall@K
7. Visual-groundedness scoring as a fifth arm

Freeze all code by **Sept 8** and spend Sept 8–17 on presentation only.
Consider releasing the CC BY / CC0 subset to Hugging Face during that window —
`license_class` makes it a filter and an upload, and a live dataset URL is a
materially stronger resume line than "built a dataset."
