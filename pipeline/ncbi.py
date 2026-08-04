"""
Rate-limited NCBI E-utilities client.

Three things this handles that the original scripts did not:

  1. RATE LIMITING that respects NCBI's published ceiling (3 req/s without an
     API key, 10 with one) using a real token spacing rather than a fixed
     time.sleep, plus exponential backoff on 429/5xx. NCBI sees a shared
     datacenter IP when you run on RunPod, so transient throttling is normal
     and must be retried, not treated as a failure.

  2. BATCHED efetch. efetch accepts many PMCIDs per call and returns one
     <pmc-articleset> containing every article. Measured on this dataset:
     20 ids in a single call takes 2.9s (0.14s/article) versus 2.26s/article
     fetched one at a time -- a ~15x speedup that also cuts your request count
     by the batch size, so the rate limit stops mattering at all.

  3. DATE-PARTITIONED search. esearch refuses retstart > 9998, so a query
     matching more than ~10k articles can never be fully enumerated in one
     go. partition_query() bisects the date range until every slice is under
     the cap, making the pool effectively unbounded.
"""

import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import requests

from . import config


class RateLimiter:
    """
    Spaces requests to at most `rate` per second, across all threads.

    The lock is held while sleeping, which is deliberate: it serializes the
    *scheduling* of requests so the global rate is honoured no matter how many
    workers are running. The requests themselves happen outside the lock, so
    throughput is unaffected -- at 10 req/s a thread holds it for at most
    0.1s, against multi-second HTTP calls.
    """

    def __init__(self, rate: float):
        self.min_interval = 1.0 / rate
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()


class NCBIClient:
    """
    Thin, polite wrapper over the E-utilities endpoints we need.

    Thread-safe: the rate limiter is shared and locked, each thread gets its
    own pooled Session, and the request counters are guarded. One client can
    be used from many workers concurrently.
    """

    def __init__(self, email=None, api_key=None, rate=None):
        self.email = email or config.EMAIL
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.limiter = RateLimiter(rate or config.RATE_LIMIT)
        self._local = threading.local()
        self._counter_lock = threading.Lock()
        self.n_requests = 0
        self.n_retries = 0

    @property
    def session(self) -> requests.Session:
        """One pooled Session per thread; `requests.Session` is not
        guaranteed thread-safe for concurrent use."""
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=4)
            s.mount("https://", adapter)
            self._local.s = s
        return s

    def _bump(self, requests_=0, retries=0) -> None:
        with self._counter_lock:
            self.n_requests += requests_
            self.n_retries += retries

    def _params(self, extra: dict) -> dict:
        p = {"email": self.email, **extra}
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def _get(self, endpoint: str, params: dict, timeout=120) -> requests.Response:
        """GET with rate limiting and exponential backoff on transient errors."""
        url = f"{config.EUTILS}/{endpoint}"
        delay = config.BACKOFF_BASE
        last_err = None
        for attempt in range(config.MAX_RETRIES):
            self.limiter.wait()
            try:
                r = self.session.get(url, params=self._params(params), timeout=timeout)
                self._bump(requests_=1)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r
            except (requests.RequestException, requests.HTTPError) as e:
                last_err = e
                self._bump(retries=1)
                if attempt == config.MAX_RETRIES - 1:
                    break
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"{endpoint} failed after {config.MAX_RETRIES} tries: {last_err}")

    # -- search ------------------------------------------------------------

    def count(self, term: str) -> int:
        """How many PMC records match, without retrieving any of them."""
        r = self._get("esearch.fcgi",
                      {"db": "pmc", "term": term, "retmax": 0, "retmode": "json"})
        return int(r.json()["esearchresult"]["count"])

    def search(self, term: str, retmax: int = 9998, retstart: int = 0) -> list:
        """Return up to `retmax` PMCIDs. retstart cannot exceed 9998."""
        if retstart > config.ESEARCH_PAGE_LIMIT:
            raise ValueError(
                f"retstart={retstart} exceeds NCBI's cap of "
                f"{config.ESEARCH_PAGE_LIMIT}; use partition_query() instead")
        r = self._get("esearch.fcgi", {
            "db": "pmc", "term": term, "retmax": retmax,
            "retstart": retstart, "retmode": "json",
        })
        res = r.json().get("esearchresult", {})
        if "ERROR" in res:
            raise RuntimeError(f"esearch error: {res['ERROR']}")
        return res.get("idlist", [])

    def search_all(self, term: str, limit=None) -> list:
        """Page through one query up to the 9998 cap, deduping as we go."""
        out, seen = [], set()
        start = 0
        while start <= config.ESEARCH_PAGE_LIMIT:
            page = self.search(term, retmax=min(500, config.ESEARCH_PAGE_LIMIT - start + 1),
                               retstart=start)
            if not page:
                break
            for pmcid in page:
                if pmcid not in seen:
                    seen.add(pmcid)
                    out.append(pmcid)
                    if limit and len(out) >= limit:
                        return out
            start += len(page)
        return out

    # -- date partitioning -------------------------------------------------

    def partition_query(self, term: str, date_from: str, date_to: str,
                        cap=None, _depth=0) -> list:
        """
        Split `term` into date slices that each match fewer than `cap` records.

        Returns a list of (sub_term, count) pairs whose union is the full
        result set. Bisects recursively, so a query matching 1M articles
        becomes however many slices are needed rather than being truncated at
        9,999.
        """
        cap = cap or config.ESEARCH_PAGE_LIMIT
        sub = f'{term} AND ("{date_from}"[PDAT] : "{date_to}"[PDAT])'
        n = self.count(sub)
        if n == 0:
            return []
        lo, hi = _parse_date(date_from), _parse_date(date_to)
        if n < cap or lo >= hi:
            # Either it fits, or we cannot split further (single day).
            return [(sub, n)]
        mid = lo + (hi - lo) / 2
        return (self.partition_query(term, date_from, _fmt(mid), cap, _depth + 1)
                + self.partition_query(term, _fmt(mid + timedelta(days=1)), date_to,
                                       cap, _depth + 1))

    # -- fetch -------------------------------------------------------------

    def fetch_batch(self, pmcids: list) -> str:
        """Fetch many articles' full-text XML in ONE request."""
        r = self._get("efetch.fcgi", {
            "db": "pmc", "id": ",".join(pmcids), "retmode": "xml",
        }, timeout=300)
        return r.text


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse_date(s: str) -> date:
    y, m, d = (int(x) for x in s.split("/"))
    return date(y, m, d)


def _fmt(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def split_articleset(xml_text: str):
    """
    Split a batched <pmc-articleset> into (pmcid, article_xml_string) pairs.

    The PMCID is read from each article's own <article-id pub-id-type="pmc">
    rather than inferred from request order. NCBI silently drops articles it
    cannot serve, so position in the response does NOT reliably correspond to
    position in the request -- assuming it does would misattribute captions
    and licenses to the wrong papers.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    articles = root.findall(".//article") or ([root] if root.tag == "article" else [])
    for art in articles:
        pmcid = article_pmcid(art)
        if pmcid:
            out.append((pmcid, ET.tostring(art, encoding="unicode")))
    return out


# JATS tags the PMCID as pub-id-type="pmcid" (value "PMC3219872") or
# "pmcaid" (value "3219872"). It is NOT "pmc" -- assuming that silently
# yields zero articles from an otherwise perfectly good response.
_PMCID_TYPES = ("pmcid", "pmcaid", "pmc")


def article_pmcid(art) -> str:
    """Numeric PMCID for one <article>, or '' if absent."""
    found = {}
    for aid in art.iter("article-id"):
        t = aid.get("pub-id-type")
        if t in _PMCID_TYPES and t not in found:
            found[t] = (aid.text or "").strip()
    for t in _PMCID_TYPES:
        if found.get(t):
            return found[t].upper().replace("PMC", "").strip()
    return ""


def article_s3_prefix(art) -> str:
    """
    Versioned S3 folder for one <article>, e.g. 'PMC3219872.2'.

    JATS carries this as pub-id-type="pmcid-ver". Using it makes the image key
    exact, so the downloader never has to guess a version or pay for a bucket
    listing. Articles revised after publication live at .2/.3, so guessing .1
    is wrong often enough to matter.
    """
    for aid in art.iter("article-id"):
        if aid.get("pub-id-type") == "pmcid-ver":
            v = (aid.text or "").strip()
            if v:
                return v.upper() if v.upper().startswith("PMC") else f"PMC{v}"
    return ""
