# Running the dataset pipeline on RunPod

Everything large — images, cached XML, manifests — is written under
`LABGEMMA_DATA`. Nothing goes on your local disk.

## 1. Pod setup

Use a **network volume**, not container disk. Container disk is wiped when the
pod stops; a network volume persists and can be re-attached to a GPU pod later
for training.

- Volume: 20 GB is plenty (5,000 pairs ≈ 0.7 GB; 100,000 ≈ 13 GB)
- Mount point: `/workspace`
- Pod type: **a CPU pod is enough for stages 1–3.** This work is network-bound,
  not GPU-bound — renting an A100 to download JPEGs wastes most of the spend.
  Attach the same volume to a GPU pod when you get to training.

```bash
git clone <your-repo> /workspace/LabGemma
cd /workspace/LabGemma
pip install requests

export LABGEMMA_DATA=/workspace/labgemma
export NCBI_EMAIL=yyang784@wisc.edu
export NCBI_API_KEY=...        # free, see below
```

Get an API key at <https://account.ncbi.nlm.nih.gov/settings/> — it raises the
rate limit from 3 to 10 requests/sec. Less critical than it sounds now that
`efetch` calls are batched 50 articles at a time, but it's free and it helps
when NCBI is throttling the shared datacenter IP.

Confirm where data will land before a long run:

```bash
python -m pipeline.config
```

## 2. Run the three stages

```bash
python -m pipeline.extract --max-articles 3000     # -> manifest.csv
python -m pipeline.download                        # -> images/ + download_log.csv
python -m pipeline.dataset --tiers high            # -> pairs.csv + stats.json
```

All three are **resumable**. If a pod dies or you Ctrl-C, re-run the same
command: `extract` skips articles already in the XML cache, `download` skips
images already on disk, and `dataset` is pure recomputation.

`download` exits non-zero if any image failed, and prints the download success
rate. Don't build a dataset on a partial download — re-run until it's clean, or
inspect `download_log.csv` to see why rows are failing.

## 3. Expected cost

Measured on a home connection; RunPod should be substantially faster on the S3
transfers, which dominate.

| Stage | 3,000 articles | Notes |
|---|---|---|
| extract | ~7 min | 0.14 s/article batched, vs 2.26 s serial |
| download | ~15 min at 16 workers | scales with `--workers` |
| dataset | seconds | pure CPU |

Disk: ~0.7 GB images + ~0.4 GB gzipped XML cache.

The XML cache is the reason a re-run is cheap. Delete `xml_cache/` if you want
to reclaim space once the manifest is stable.

## 4. Scaling knobs

Everything is in [pipeline/config.py](pipeline/config.py):

- `SEARCH_TERM` — currently a narrow Title/Abstract query matching 77,350
  articles. A broad all-fields variant matches over 1M but with a lower
  gel-figure density per article.
- `DATE_FROM` / `DATE_TO` — `extract` automatically bisects this range into
  slices under NCBI's 9,998-record `retstart` cap, so the search pool is not
  limited to the first 9,999 hits.
- `EFETCH_BATCH` (50) and `S3_WORKERS` (16).
- Tier keyword lists and `SPLIT_FRACTIONS`.

## 5. Filtering for the experiments

The plan's controlled experiments come out of stage 3 flags — the manifest is
collected once and filtered many ways, so you never re-download to change an
experimental arm:

```bash
python -m pipeline.dataset --tiers high                    # high-confidence only
python -m pipeline.dataset --tiers high medium             # + medium
python -m pipeline.dataset --tiers high --licenses "CC BY" CC0   # commercial-safe
python -m pipeline.dataset --tiers high --drop-compound    # single-panel only
```
