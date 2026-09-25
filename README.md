# LabGemma

**What does a multimodal model actually use when it matches a biomedical figure to its caption?**

On same-paper hard negatives, zero-shot Gemma 3 4B reaches **0.896 AUROC**. Masking the text
rendered inside the figure drops it to **0.745**; an OCR-and-symbol-overlap baseline reaches only
**0.622**; chance is **0.504**.

So **~38% of the model's above-chance performance comes from text printed inside the figure, and
~62% from visual content** — and with the text removed, it still beats a system whose entire
strategy is reading that text.

## Results

Figure–caption matching on 1,596 held-out gel/blot figures from PubMed Central. **Hard negatives
are a different figure from the same paper**, so "which paper is this?" carries no signal.

| AUROC | overall | easy neg | **hard neg** | compound | single |
|---|---|---|---|---|---|
| Random | 0.493 | 0.485 | 0.504 | — | — |
| OCR symbol overlap | 0.724 | 0.808 | 0.622 | — | — |
| Zero-shot, text masked | 0.832 | 0.903 | 0.745 | 0.834 | 0.828 |
| Zero-shot | 0.947 | 0.989 | **0.896** | 0.949 | 0.941 |
| LoRA a1 (easy negs), masked | 0.904 | 0.993 | 0.795 | 0.908 | 0.893 |
| LoRA a1 (easy negs) | 0.965 | 0.999 | **0.922** | 0.972 | 0.948 |
| LoRA a2 (hard negs), masked | 0.845 | 0.913 | 0.762 | 0.847 | 0.841 |
| LoRA a2 (hard negs) | 0.956 | 0.992 | **0.911** | 0.957 | 0.950 |

*n = 4,496 test rows. AUROC standard error ≈ 0.006, so differences under ~0.012 are not decisive.*

**It is not name matching.** Protein symbols are printed beside the bands and captions name the
same proteins, so "read the labels, match the strings" is a real strategy needing no biology. It
caps at 0.622. The model reaches 0.896.

**Fine-tuning barely moves ranking** (0.896 → 0.922), but **hard negatives fix calibration** — the
clearest effect in the study, and invisible in AUROC:

| hard negatives | AUROC | accuracy |
|---|---|---|
| Zero-shot | 0.896 | 0.774 |
| a1 (easy negatives) | 0.922 | **0.713** |
| a2 (hard negatives) | 0.911 | **0.823** |

a1 ranks better while classifying worse. Trained only on obvious negatives, it learned that
non-matches announce themselves, so on same-paper negatives it says "yes" too readily. a2 learned
where the boundary is: **+11 accuracy points** at equal AUROC. Both arms train on identical
positives and identical row counts (13,874) — only negative type differs.

**Panel layout contributes nothing.** 72% of figures are compound, and a caption enumerating
(A)–(D) against a four-panel image is matchable without reading. Compound and single score within
0.008, and drop identically under masking.

**Two predictions failed and are recorded as such.** Hard-negative training did *not* reduce text
dependence (38% zero-shot, 36% a2, 30% a1). And caption "visual anchor" density did not predict
success — because the anchors are generic: *western blot*, *kDa*, *GAPDH* appear in **both**
captions of a same-paper pair. What discriminates is protein symbols (median Jaccard 0.31 between
same-paper captions).

## Where it fails

Error rates for a2 at threshold 0.5 (12.3% overall). Only two comparisons survive two-proportion
significance testing:

| group | n | error | |
|---|---|---|---|
| Positives | 1,596 | 4.8% | |
| Easy negatives | 1,596 | 2.5% | |
| **Hard negatives** | 1,304 | **33.4%** | 22.7 SE vs easy |
| **Captions >224 words** | 1,116 | **15.3%** | 3.6 SE vs medium |
| Compound vs single | — | 12.1% / 12.7% | 0.5 SE — no effect |

Essentially all error is on same-paper negatives — a 13× rate difference. The task is easy except
in exactly the condition the benchmark was built to make hard.

## Why this question

PMC figure–caption pairs are the standard weak-supervision signal in biomedical VLP (PMC-CLIP,
MICCAI 2023; LLaVA-Med, NeurIPS 2023; BiomedCLIP, NEJM AI; Open-PMC-18M, CVPR) at 1.6M–18M pairs
spanning every imaging modality at once. But biomedical captions describe *experimental context*,
not appearance: measured here, only 22% of caption sentences contain anything visually checkable.

Restricting to one assay type makes the decomposition answerable — in gel/blot figures the text
channel is well defined and the visual content uniform. Same-paper hard negatives are the
biomedical analogue of the complementary image pairs in Goyal et al., *Making the V in VQA Matter*
(CVPR 2017); the masking ablation follows work isolating OCR pathways in VLMs (EMNLP 2025).

## Dataset

| | |
|---|---|
| Search pool | 928,659 articles |
| Fetched | 103,000 articles in 18.8 min |
| Figures extracted | **202,224** across 73,691 articles |
| CC-licensed | 175,949 (87%) — CC BY alone 83,092 |
| Used | **9,894 pairs** / 4,522 articles |
| Split (by paper) | 6,937 / 1,361 / 1,596 — **0 article leakage** |
| Compound figures | 72.1% |

Splits are hashed from PMCID, so no article spans two splits and an article's split never changes
as the dataset grows. See [DATASET_CARD.md](DATASET_CARD.md).

## Method

```bash
python -m pipeline.extract  --max-articles 103000
python -m pipeline.download --tiers high --licenses "CC BY" "CC BY-NC" CC0 --limit 10000
python -m pipeline.dataset  --tiers high --licenses "CC BY" "CC BY-NC" CC0
python -m pipeline.task
python -m pipeline.ocr extract && python -m pipeline.baselines
python -m pipeline.ocr mask --split test
python -m pipeline.train score --out preds_zeroshot.json
python -m pipeline.train fit --arm a2 && python -m pipeline.train score --adapter runs/a2 --out preds_a2.json
python -m pipeline.report preds_*.json
```

Three decisions worth naming:

- **Scoring is a logit ratio**, `P(yes)/(P(yes)+P(no))`, not generation — a discrete string
  discards confidence and collapses AUROC to accuracy.
- **Masking uses every OCR detection while extraction trusts only confident ones.** At a shared
  threshold, text that read weakly survived masking and became legible once its neighbours were
  painted out; masked images still scored 0.620. Asymmetric thresholds drop symbol-bearing images
  from 79% to 23% and the OCR baseline to 0.537.
- **Youden's J, not F1**, for the operating point. F1 ignores true negatives, so with 55% of
  overlap scores tied at 0 it is maximised by predicting everything positive — reporting accuracy
  0.355, *below* the random baseline.

## Limitations

The 38% has bias both ways: 23% of masked images still yield OCR symbols (pushing it down), but
box-filling removes image area as well as glyphs (pushing it up). Treat it as an order of
magnitude.

Positives are weak labels — co-publication, unverified, and 72% compound. Negatives can be
semantically true ("GAPDH was used as a loading control" describes thousands of these blots).
Contamination is unmeasured; the pipeline supports a pre/post-cutoff split but it was not run.
One model, one scale.

## Not done

Arm A3 (high+medium tiers — the weak-label-quality axis), arm A4 (`--drop-compound`), a BiomedCLIP
baseline, and the contamination test.

## Repository

| | |
|---|---|
| `pipeline/extract.py` | Date-partitioned PMC search, batched efetch, JATS parsing, license classification |
| `pipeline/scoring.py` | Confidence tiers, caption quality, compound-figure detection |
| `pipeline/download.py` | Parallel S3 fetch, idempotent, article-level sampling |
| `pipeline/dataset.py` | Dedup, paper-level splitting, statistics |
| `pipeline/task.py` | Positives, easy and same-paper hard negatives, frozen eval sets |
| `pipeline/ocr.py` | Tesseract wrapper, symbol extraction, text masking |
| `pipeline/baselines.py` | Random and OCR baselines, AUROC, coverage |
| `pipeline/train.py` | LoRA fit, logit-ratio scoring, prediction probe |
| `pipeline/report.py` | Run comparison and breakdowns |
| `pipeline/errors.py` | Failure analysis |

Setup: [RUNPOD.md](RUNPOD.md). Data never enters the repo; everything large lives under
`$LABGEMMA_DATA`.
