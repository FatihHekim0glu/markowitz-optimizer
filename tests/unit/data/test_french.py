"""Tests for the Ken French factor loader."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from markowitz.data.exceptions import (
    DataIntegrityError,
    EmptyDataError,
    ProviderUnavailableError,
)
from markowitz.data.french import (
    _build_url,
    _parse_french_csv,
    load_french_factors,
)

_SAMPLE_MONTHLY_CSV = """\
This file was created by Kenneth French
Some preamble line
,Mkt-RF,SMB,HML,RF
192607,2.96,-2.30,-2.87,0.22
192608,2.64,-1.40,4.19,0.25
192609,0.36,-1.32,0.01,0.23
"""


_SAMPLE_DAILY_CSV = """\
Daily factors
,Mkt-RF,SMB,HML,RF
19260701,0.10,0.05,-0.02,0.01
19260702,-0.20,0.07,0.03,0.01
"""


def test_french_parser_units_monthly() -> None:
    frame = _parse_french_csv(_SAMPLE_MONTHLY_CSV, frequency="monthly")
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    # 2.96 in percent -> 0.0296 in decimal
    assert frame.iloc[0]["Mkt-RF"] == pytest.approx(0.0296)
    assert frame.iloc[0]["RF"] == pytest.approx(0.0022)


def test_french_parser_divides_by_100_once() -> None:
    frame = _parse_french_csv(_SAMPLE_MONTHLY_CSV, frequency="monthly")
    # Values should be in low double-digit percent range, i.e. |x| < 1.
    assert frame.abs().max().max() < 1.0
    # And not divided twice (e.g. 0.000296)
    assert frame.iloc[0]["Mkt-RF"] > 0.001


def test_french_parser_daily_index() -> None:
    frame = _parse_french_csv(_SAMPLE_DAILY_CSV, frequency="daily")
    assert frame.index[0] == pd.Timestamp("1926-07-01")
    assert frame.index[1] == pd.Timestamp("1926-07-02")


def test_french_parser_missing_header_raises() -> None:
    bad = "garbage with no leading-empty header row"
    with pytest.raises(DataIntegrityError, match="header"):
        _parse_french_csv(bad, frequency="monthly")


def test_french_parser_no_rows_raises() -> None:
    just_header = ",Mkt-RF\n"
    with pytest.raises(EmptyDataError):
        _parse_french_csv(just_header, frequency="monthly")


def test_build_url_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="unknown frequency"):
        _build_url("F-F_Research_Data_Factors", "hourly")


def test_load_french_factors_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build an in-memory zip mimicking the French archive
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors.CSV", _SAMPLE_MONTHLY_CSV)
    payload = buf.getvalue()

    cache_file = tmp_path / "F-F_Research_Data_Factors_monthly.zip"
    cache_file.write_bytes(payload)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not be touched when cache is hot")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    frame = load_french_factors(
        "F-F_Research_Data_Factors", frequency="monthly", cache_root=tmp_path
    )
    assert "Mkt-RF" in frame.columns


def test_load_french_factors_network_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ProviderUnavailableError, match="unreachable"):
        load_french_factors("F-F_Research_Data_Factors", frequency="monthly", cache_root=tmp_path)


def test_load_french_factors_bad_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_file = tmp_path / "F-F_Research_Data_Factors_monthly.zip"
    cache_file.write_bytes(b"definitely not a zip")
    with pytest.raises(DataIntegrityError, match="corrupt"):
        load_french_factors("F-F_Research_Data_Factors", frequency="monthly", cache_root=tmp_path)


def test_load_french_factors_downloads_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors.CSV", _SAMPLE_MONTHLY_CSV)
    payload = buf.getvalue()

    class _FakeResp:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def read(self) -> bytes:
            return self._data

    def fake_open(_url: str, **_kwargs: Any) -> _FakeResp:
        return _FakeResp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    frame = load_french_factors(
        "F-F_Research_Data_Factors", frequency="monthly", cache_root=tmp_path
    )
    assert "Mkt-RF" in frame.columns

    # Cache file should now exist
    cache_file = tmp_path / "F-F_Research_Data_Factors_monthly.zip"
    assert cache_file.exists()
    assert cache_file.read_bytes() == payload


def test_load_french_factors_no_csv_in_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no csv here")
    cache_file = tmp_path / "X_monthly.zip"
    cache_file.write_bytes(buf.getvalue())
    with pytest.raises(DataIntegrityError, match="no CSV"):
        load_french_factors("X", frequency="monthly", cache_root=tmp_path)


_CSV_WITH_BLANK_BREAK = """\
Header preamble
,Mkt-RF,SMB,HML,RF
192607,2.96,-2.30,-2.87,0.22
192608,2.64,-1.40,4.19,0.25

192700,99.99,99.99,99.99,99.99
"""


_CSV_WITH_NON_NUMERIC_BREAK = """\
Header preamble
,Mkt-RF,SMB,HML,RF
192607,2.96,-2.30,-2.87,0.22
192608,2.64,-1.40,4.19,0.25
Annual Factors: January-December,,,,
192700,99.99,99.99,99.99,99.99
"""


def test_french_parser_stops_on_blank_row_within_data() -> None:
    frame = _parse_french_csv(_CSV_WITH_BLANK_BREAK, frequency="monthly")
    # The annual block after the blank line must not be ingested.
    assert len(frame) == 2
    assert frame.iloc[-1]["Mkt-RF"] == pytest.approx(0.0264)


def test_french_parser_stops_on_non_numeric_row() -> None:
    frame = _parse_french_csv(_CSV_WITH_NON_NUMERIC_BREAK, frequency="monthly")
    # Parser must stop at the "Annual Factors" sentinel.
    assert len(frame) == 2
    assert frame.iloc[-1]["Mkt-RF"] == pytest.approx(0.0264)


def test_load_french_factors_generic_exception_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Any non-URLError raised by urlopen must be wrapped as ProviderUnavailableError
    # (exercises the generic ``except Exception`` branch at lines 132-133).
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("socket timeout simulator")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ProviderUnavailableError, match="Failed to download"):
        load_french_factors("F-F_Research_Data_Factors", frequency="monthly")


def test_load_french_factors_without_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # cache_root=None exercises the ``cache_dir is None`` branch (117->122)
    # and the post-download ``cache_file is None`` branch (136->139).
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors.CSV", _SAMPLE_MONTHLY_CSV)
    payload = buf.getvalue()

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Resp())
    frame = load_french_factors("F-F_Research_Data_Factors", frequency="monthly")
    assert "Mkt-RF" in frame.columns
