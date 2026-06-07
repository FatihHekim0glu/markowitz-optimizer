"""Survivorship-bias-aware S&P 500 membership builder.

A truly bias-free constituent history would require a paid index-rebalance
feed (S&P Dow Jones Indices, CRSP). We approximate it instead by intersecting
a recent snapshot of the index members with the Polygon grouped-daily snapshot
on each as-of date: a ticker counts as a member iff it appears in
:data:`CURRENT_SP500` AND has a real trading bar on the requested day.

This is materially better than the naive "today's-list on yesterday's-date"
approach because:

- Symbols that had not yet IPO'd by the as-of date drop out (no grouped-daily
  row), which prevents look-ahead leakage from the modern constituent list.
- Every returned ticker is guaranteed to have same-day OHLCV available, which
  is the dominant correctness concern in walk-forward backtests.

Known limitations
-----------------
- Companies that were once in the index but have since been delisted or
  acquired (Lehman, EMC, Sprint, ...) are missing. That is the pure
  "survivor" blind spot and biases backtests upward on average.
- Modern names added to the index after they had been trading publicly
  (e.g. mid-2010s tech IPOs) are over-included before their real entry date.

When no Polygon provider is supplied the builder emits a warning and returns
the static :data:`CURRENT_SP500` list as-is — that path is explicitly
survivorship-biased and should only be used for offline demos.

Caching
-------
Membership lists are cached in-memory keyed by the as-of date. There is no
DB write — long-running services that need persistence should layer their
own store on top.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .polygon import PolygonProvider
    from .yfinance import YFinanceProvider

logger = logging.getLogger(__name__)


# Static snapshot of S&P 500 tickers as of mid-2024 (source: datahub.io
# core/s-and-p-500-companies, a Wikipedia mirror). Treated as the "modern
# constituents" set and intersected with Polygon grouped-daily to approximate
# point-in-time membership.
CURRENT_SP500: tuple[str, ...] = (
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT",
    "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL",
    "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK",
    "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BK", "BBWI",
    "BAX", "BDX", "WRB", "BRK-B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BX",
    "BA", "BKNG", "BWA", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR",
    "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX",
    "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "COR", "CNC",
    "CNP", "CF", "CHRD", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD",
    "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO",
    "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT",
    "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI",
    "CVS", "DHR", "DRI", "DVA", "DAY", "DECK", "DE", "DAL", "DVN", "DXCM",
    "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DHI",
    "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA",
    "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX",
    "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD",
    "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
    "FSLR", "FE", "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN",
    "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS",
    "GM", "GPC", "GILD", "GPN", "GL", "GS", "HAL", "HIG", "HAS", "HCA",
    "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL",
    "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX",
    "ITW", "ILMN", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG",
    "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J",
    "JNJ", "JCI", "JPM", "JNPR", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB",
    "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW",
    "LVS", "LDOS", "LEN", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LULU",
    "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH",
    "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP",
    "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST",
    "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA",
    "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG",
    "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE",
    "ORCL", "OTIS", "PCAR", "PKG", "PANW", "PARA", "PH", "PAYX", "PAYC", "PYPL",
    "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "POOL", "PPG",
    "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM",
    "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF",
    "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM",
    "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA",
    "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI",
    "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL",
    "TDY", "TFX", "TER", "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TSCO",
    "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR",
    "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO",
    "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "V", "VICI", "WAB", "WBA", "WMT",
    "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WY",
    "WHR", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
)


class SP500UniverseBuilder:
    """Builds and caches point-in-time S&P 500 membership snapshots.

    Parameters
    ----------
    provider:
        Either a :class:`PolygonProvider` (preferred — enables PIT membership
        via grouped-daily) or a :class:`YFinanceProvider` (fallback — emits a
        warning and returns the survivorship-biased static list).
    """

    def __init__(
        self,
        provider: PolygonProvider | YFinanceProvider | None = None,
    ) -> None:
        self._provider = provider
        self._cache: dict[date, list[str]] = {}
        self._warned_fallback = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_membership_as_of(self, date_: date) -> list[str]:
        """Return the approximated S&P 500 membership on ``date_``.

        When the configured provider exposes a working ``get_grouped_daily``
        (Polygon path), the result is the intersection of :data:`CURRENT_SP500`
        with the symbols that actually traded on ``date_``. When it does not
        (no provider, yfinance fallback, or grouped-daily empty), the static
        list is returned and a warning is emitted on the first such call.
        """
        cached = self._cache.get(date_)
        if cached is not None:
            return list(cached)

        if self._provider is None:
            members = self._fallback_members()
        else:
            try:
                grouped = self._provider.get_grouped_daily(date_)
            except Exception as exc:
                logger.warning(
                    "grouped-daily fetch failed for %s (%s); using static list",
                    date_,
                    exc,
                )
                members = self._fallback_members()
            else:
                if grouped.empty:
                    logger.warning(
                        "grouped-daily empty for %s; returning empty membership",
                        date_,
                    )
                    members = []
                else:
                    active = set(grouped.index)
                    members = sorted(t for t in CURRENT_SP500 if t in active)

        self._cache[date_] = list(members)
        return list(members)

    def get_membership_window(
        self,
        start: date,
        end: date,
        freq: str = "ME",
    ) -> dict[date, list[str]]:
        """Build membership at each rebalance date in ``[start, end]``.

        ``freq`` follows pandas offset aliases; default ``ME`` = month-end,
        matching the cadence used by most monthly walk-forward backtests.
        When ``[start, end]`` is shorter than one period the window degenerates
        to ``{start, end}`` so callers always get at least two anchors back.
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        anchors = pd.date_range(start=start, end=end, freq=freq)
        if len(anchors) == 0:
            anchors = pd.DatetimeIndex(
                [pd.Timestamp(start), pd.Timestamp(end)]
            ).unique()
        result: dict[date, list[str]] = {}
        for ts in anchors:
            d = ts.date() if hasattr(ts, "date") else ts
            result[d] = self.get_membership_as_of(d)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fallback_members(self) -> list[str]:
        if not self._warned_fallback:
            warnings.warn(
                "S&P 500 universe falling back to the static today-list — "
                "this path is survivorship-biased; configure POLYGON_API_KEY "
                "for point-in-time membership.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_fallback = True
        return sorted(CURRENT_SP500)
