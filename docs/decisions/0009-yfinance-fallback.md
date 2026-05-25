# ADR-0009: yfinance primary, Tiingo fallback for price data

* Status: Accepted
* Date: 2026-05-23

## Context

The data layer needs a free-tier source for end-of-day equity prices. yfinance is convenient but rate-limited and occasionally returns malformed data. Paid providers (Bloomberg, Refinitiv) are out of scope.

## Decision

Primary: **yfinance** (free, no API key). Fallback: **Tiingo** (free tier, 250 requests/day, requires API key in `TIINGO_API_KEY` env var). Fallback engages only on `RateLimitError` or `ProviderUnavailableError` from the primary, never on `EmptyDataError` (an empty result is a real answer).

## Decision drivers

- Cost: zero by default.
- Resilience: yfinance rate limits would otherwise break heavy backtest development.
- Convention: cassette-replay testing (`vcrpy`) makes tests deterministic without network.

## Considered options

- Option A: yfinance primary + Tiingo fallback. **Chosen.**
- Option B: yfinance only. Rejected: rate-limit risk during dev.
- Option C: Pin to a paid provider. Rejected: cost, accessibility.

## Consequences

- The `PriceProvider` Protocol allows swapping providers without touching callers.
- `CachedProvider` wraps any provider with Parquet caching (24h TTL by default).
- Network tests are marked `@pytest.mark.network` and excluded from default CI.
- Ken French datasets are fetched directly via HTTP (separate path from price providers).

## Links

- yfinance: https://github.com/ranaroussi/yfinance
- Tiingo: https://www.tiingo.com
