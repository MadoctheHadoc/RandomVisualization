#!/usr/bin/env python3
"""
Polymarket Calibration Data Collector
=====================================
Fetches resolved binary markets from Polymarket's public APIs and builds
a calibration dataset: the implied probability at a chosen snapshot time
before resolution vs. whether the market resolved YES.

Output
------
    data/polymarket_raw.json       — cached raw API responses (skipped on re-run)
    data/calibration_data.csv      — per-contract rows (implied_prob, resolved, …)
    data/calibration_buckets.csv   — pre-computed bucket stats with Wilson 95% CIs

Drop calibration_buckets.csv straight into the calibration visualiser.

Usage
-----
    python collect_calibration_data.py --days-before 7
    python collect_calibration_data.py --days-before 7 --force-fetch   # ignore cache
    python collect_calibration_data.py --debug                         # inspect raw fields

Dependencies
------------
    pip install requests tqdm
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB  = "https://clob.polymarket.com"
POLY_SLEEP       = 0.05
DATA_DIR         = Path("data")

# ---------------------------------------------------------------------------
# Bucket definitions: 1 % at extremes, 5 % in the middle
# ---------------------------------------------------------------------------
BUCKETS = (
    [(i, i + 1) for i in range(0, 5)]          # 0-1, 1-2, 2-3, 3-4, 4-5
    + [(i, i + 5) for i in range(5, 95, 5)]    # 5-10, 10-15, … 90-95
    + [(i, i + 1) for i in range(95, 100)]      # 95-96, 96-97, 97-98, 98-99, 99-100
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url, params=None, headers=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 2 ** (attempt + 2)
                log.warning("Rate-limited; sleeping %ds", wait)
                time.sleep(wait)
            else:
                raise
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def _parse_json_field(v):
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _to_ts(iso_str):
    if not iso_str:
        return None
    s = iso_str.rstrip("Z").replace("+00:00", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(s, fmt)
                       .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half   = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n
                                      + z * z / (4 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return p_hat, lo, hi


# ---------------------------------------------------------------------------
# Polymarket collector
# ---------------------------------------------------------------------------

class PolymarketCollector:
    def __init__(self, days_before=7, max_markets=1000,
                 tags=None, min_volume=100, debug=False):
        self.days_before = days_before
        self.max_markets = max_markets
        self.tags        = [t.lower() for t in tags] if tags else None
        self.min_volume  = min_volume
        self.debug       = debug
        self._reject     = Counter()

    def _gamma_pages(self):
        offset, limit, fetched = 0, 100, 0
        log.info("[Polymarket] Fetching resolved markets from Gamma API…")
        while fetched < self.max_markets:
            batch = min(limit, self.max_markets - fetched)
            params = {"closed": "true", "limit": batch,
                      "offset": offset, "order": "volume",
                      "ascending": "false"}
            try:
                data = _get(f"{POLYMARKET_GAMMA}/markets", params=params)
            except Exception as e:
                log.error("[Polymarket] Gamma API error: %s", e)
                break
            markets = data if isinstance(data, list) else data.get("markets", [])
            if not markets:
                break
            yield from markets
            fetched += len(markets)
            offset  += len(markets)
            if len(markets) < batch:
                break
            time.sleep(POLY_SLEEP)

    def _clob_price(self, token_id, snapshot_ts):
        for window_days in (1, 3, 7, 14):
            params = {"market": token_id,
                      "startTs": snapshot_ts - window_days * 86_400,
                      "endTs": snapshot_ts + 86_400,
                      "fidelity": 60}
            try:
                data = _get(f"{POLYMARKET_CLOB}/prices-history", params=params)
            except Exception:
                time.sleep(POLY_SLEEP)
                continue
            history = data.get("history", [])
            if not history:
                time.sleep(POLY_SLEEP)
                continue
            best  = min(history, key=lambda c: abs(c.get("t", 0) - snapshot_ts))
            price = best.get("c") or best.get("p")
            if price is not None:
                return float(price)
        return None

    def _process(self, m):
        if self.tags:
            market_tags = [t.get("label", "").lower()
                           for t in (m.get("tags") or [])]
            if not any(t in market_tags for t in self.tags):
                self._reject["tag_mismatch"] += 1
                return None

        outcomes = _parse_json_field(m.get("outcomes", []))
        if len(outcomes) != 2:
            self._reject[f"outcomes_{len(outcomes)}"] += 1
            return None

        resolved = None
        res_legacy = m.get("resolutionPrice") or m.get("resolvedPrice")
        if res_legacy is not None:
            try:
                resolved = 1 if float(res_legacy) >= 0.5 else 0
            except (TypeError, ValueError):
                pass
        if resolved is None:
            outcome_prices = _parse_json_field(m.get("outcomePrices", []))
            if len(outcome_prices) >= 2:
                try:
                    p0 = float(outcome_prices[0])
                    if p0 >= 0.99:
                        resolved = 1
                    elif p0 <= 0.01:
                        resolved = 0
                except (TypeError, ValueError):
                    pass
        if resolved is None:
            if m.get("umaResolutionStatus") == "resolved":
                self._reject["resolved_but_ambiguous_price"] += 1
            else:
                self._reject["not_resolved"] += 1
            return None

        is_closed = (m.get("closed") is True
                     or m.get("umaResolutionStatus") == "resolved"
                     or m.get("automaticallyResolved") is True)
        if not is_closed:
            self._reject["still_open"] += 1
            return None

        volume = 0.0
        for vf in ("volumeNum", "volume", "volumeClob", "liquidityNum"):
            v = m.get(vf)
            if v is not None:
                try:
                    volume = float(v)
                    if volume > 0:
                        break
                except (TypeError, ValueError):
                    pass
        if volume < self.min_volume:
            self._reject["low_volume"] += 1
            return None

        end_ts = (_to_ts(m.get("closedTime"))
                  or _to_ts(m.get("umaEndDate"))
                  or _to_ts(m.get("endDate"))
                  or _to_ts(m.get("endDateIso")))
        if end_ts is None:
            self._reject["no_end_date"] += 1
            return None
        snapshot_ts = end_ts - self.days_before * 86_400

        clob_ids = _parse_json_field(m.get("clobTokenIds", []))
        if not clob_ids:
            cid = m.get("conditionId") or m.get("questionID")
            if cid:
                clob_ids = [cid]
            else:
                self._reject["no_clob_token"] += 1
                return None

        yes_token = clob_ids[0]
        time.sleep(POLY_SLEEP)
        price = self._clob_price(yes_token, snapshot_ts)
        if price is None:
            self._reject["no_price_history"] += 1
            return None

        return {
            "source":        "polymarket",
            "market_id":     m.get("id", m.get("conditionId", "")),
            "title":         (m.get("question") or m.get("title") or "")[:120],
            "volume_usd":    round(volume, 2),
            "snapshot_date": datetime.fromtimestamp(
                snapshot_ts, tz=timezone.utc).date().isoformat(),
            "implied_prob":  round(price, 6),
            "resolved":      resolved,
        }

    def fetch_raw(self, force=False):
        """Return raw market dicts, using cache when available."""
        cache_path = DATA_DIR / "polymarket_raw.json"
        if not force and cache_path.exists():
            log.info("[Polymarket] Loading cached raw data from %s", cache_path)
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        markets = list(self._gamma_pages())
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(markets, f)
        log.info("[Polymarket] Cached %d raw markets → %s",
                 len(markets), cache_path)
        return markets

    def collect(self, force_fetch=False):
        markets = self.fetch_raw(force=force_fetch)
        if self.debug and markets:
            log.info("[Polymarket] DEBUG — field names in first market:")
            for k, v in markets[0].items():
                log.info("  %-35s %s", k, str(v)[:80])
        rows = []
        for m in tqdm(markets, desc="Polymarket markets", unit="mkt"):
            try:
                row = self._process(m)
            except Exception as e:
                log.debug("[Polymarket] exception on %s: %s",
                          m.get("id", "?"), e)
                self._reject["exception"] += 1
                continue
            if row:
                rows.append(row)
        log.info("[Polymarket] Collected %d rows.  Rejection reasons: %s",
                 len(rows), dict(self._reject.most_common()))
        return rows


# ---------------------------------------------------------------------------
# Bucket computation with Wilson CIs
# ---------------------------------------------------------------------------

def bucketize(rows):
    """
    Assign each contract to a probability bucket and compute:
      - actual resolution rate (p_hat)
      - Wilson 95 % confidence interval (ci_lower, ci_upper)
    """
    results = []
    for lo, hi in BUCKETS:
        subset = [r for r in rows
                  if (lo == 0 and r["implied_prob"] * 100 >= 0
                      and r["implied_prob"] * 100 < hi)
                  or (lo > 0 and r["implied_prob"] * 100 >= lo
                      and r["implied_prob"] * 100 < hi)]
        n = len(subset)
        if n == 0:
            continue
        successes = sum(1 for r in subset if r["resolved"] == 1)
        actual, ci_lo, ci_hi = wilson_ci(successes, n)
        results.append({
            "bucket_lo":   lo,
            "bucket_hi":   hi,
            "n":           n,
            "mid_prob":    round((lo + hi) / 2, 2),
            "actual_rate": round(actual, 6),
            "ci_lower":    round(ci_lo, 6),
            "ci_upper":    round(ci_hi, 6),
        })
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CONTRACT_FIELDS = ["source", "market_id", "title", "volume_usd",
                   "snapshot_date", "implied_prob", "resolved"]

BUCKET_FIELDS = ["bucket_lo", "bucket_hi", "n", "mid_prob",
                 "actual_rate", "ci_lower", "ci_upper"]


def save_csv(rows, path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    log.info("Saved %d rows → %s", len(rows), path)


def print_summary(rows, buckets):
    if not rows:
        log.warning("No data collected.")
        return
    import statistics
    n        = len(rows)
    yes_rate = sum(r["resolved"] for r in rows) / n * 100
    probs    = [r["implied_prob"] for r in rows]
    resolved = [r["resolved"] for r in rows]
    brier    = sum((p - rv) ** 2 for p, rv in zip(probs, resolved)) / n
    ls       = [r for r in rows if r["implied_prob"] < 0.10]
    if ls:
        ls_bias = (sum(r["resolved"] for r in ls) / len(ls)
                   - statistics.mean(r["implied_prob"] for r in ls)) * 100
        ls_str = f"{ls_bias:+.1f}pp"
    else:
        ls_str = "N/A"

    print("\n" + "=" * 55)
    print(f"  Total contracts  : {n:,}")
    print(f"  Resolution rate  : {yes_rate:.1f}%")
    print(f"  Brier score      : {brier:.4f}")
    print(f"  Longshot bucket  : {len(ls):,}  (implied_prob < 10%)")
    print(f"  Longshot bias    : {ls_str}")
    print(f"  Buckets emitted  : {len(buckets)}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Collect Polymarket calibration data.")
    p.add_argument("--days-before", type=int, default=7,
                   help="Snapshot price N days before resolution (default: 7)")
    p.add_argument("--max-markets", type=int, default=1000)
    p.add_argument("--min-volume", type=float, default=100,
                   help="Minimum USD volume to include (default: 100)")
    p.add_argument("--tags", nargs="+", default=None,
                   help="Tag filter, e.g. --tags politics sports")
    p.add_argument("--force-fetch", action="store_true",
                   help="Ignore cached polymarket_raw.json and re-fetch")
    p.add_argument("--debug", action="store_true",
                   help="Print raw field names from first market")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    collector = PolymarketCollector(
        days_before = args.days_before,
        max_markets = args.max_markets,
        tags        = args.tags,
        min_volume  = args.min_volume,
        debug       = args.debug,
    )
    rows = collector.collect(force_fetch=args.force_fetch)

    if not rows:
        log.error("No rows. Run with --debug to inspect raw API field names.")
        sys.exit(1)

    buckets = bucketize(rows)
    print_summary(rows, buckets)

    save_csv(rows, DATA_DIR / "calibration_data.csv", CONTRACT_FIELDS)
    save_csv(buckets, DATA_DIR / "calibration_buckets.csv", BUCKET_FIELDS)


if __name__ == "__main__":
    main()