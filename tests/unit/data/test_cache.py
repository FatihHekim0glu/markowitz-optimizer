"""Tests for the Parquet-backed cache layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markowitz.data import cache as cache_mod
from markowitz.data.cache import (
    CacheCorruptionWarning,
    cache_path,
    read_cache,
    write_cache,
)

pyarrow = pytest.importorskip("pyarrow")


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame({"close": np.arange(5, dtype="float64")}, index=idx)


def test_cache_path_components(tmp_path: Path) -> None:
    p = cache_path(tmp_path, "AAPL", "yfinance", "1d")
    assert p == tmp_path / "yfinance" / "1d" / "AAPL.parquet"


def test_cache_path_special_chars(tmp_path: Path) -> None:
    p = cache_path(tmp_path, "BRK.B", "yfinance", "1d")
    assert p.suffix == ".parquet"
    assert "BRK.B" in p.name  # dot preserved
    p2 = cache_path(tmp_path, "BTC-USD", "yfinance", "1d")
    assert p2.name == "BTC-USD.parquet"
    p3 = cache_path(tmp_path, "BF/B", "yfinance", "1d")
    assert "/" not in p3.name
    assert p3.name == "BF_B.parquet"


def test_cache_atomic_write(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    target = write_cache(sample_frame, tmp_path, "AAPL", "yfinance", "1d")
    assert target.exists()
    round_trip = pd.read_parquet(target)
    # Parquet drops the index frequency attribute; check values + dtypes only.
    pd.testing.assert_frame_equal(round_trip, sample_frame, check_freq=False)


def test_cache_read_roundtrip(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    write_cache(sample_frame, tmp_path, "MSFT", "yfinance", "1d")
    got = read_cache(tmp_path, "MSFT", "yfinance", "1d")
    assert got is not None
    pd.testing.assert_frame_equal(got, sample_frame, check_freq=False)


def test_cache_read_miss_returns_none(tmp_path: Path) -> None:
    assert read_cache(tmp_path, "NOPE", "yfinance", "1d") is None


def test_cache_read_corrupt_returns_none_warns(tmp_path: Path) -> None:
    p = cache_path(tmp_path, "JUNK", "yfinance", "1d")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"this is not a parquet file")
    with pytest.warns(CacheCorruptionWarning):
        result = read_cache(tmp_path, "JUNK", "yfinance", "1d")
    assert result is None


def test_cache_no_tempfiles_left_behind(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    write_cache(sample_frame, tmp_path, "AAPL", "yfinance", "1d")
    target_dir = tmp_path / "yfinance" / "1d"
    leftover = [p for p in target_dir.iterdir() if p.suffix == ".tmp"]
    assert leftover == []


def test_cache_write_creates_parent_dirs(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    assert not (tmp_path / "alpha").exists()
    write_cache(sample_frame, tmp_path, "X", "alpha", "1d")
    assert (tmp_path / "alpha" / "1d" / "X.parquet").exists()


def test_cache_write_cleanup_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_frame: pd.DataFrame
) -> None:
    real_to_parquet = pd.DataFrame.to_parquet

    def boom(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        write_cache(sample_frame, tmp_path, "X", "src", "1d")

    # restore so other tests are unaffected
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real_to_parquet)

    # The target dir might exist (created during write); no leftover .tmp
    target_dir = tmp_path / "src" / "1d"
    if target_dir.exists():
        leftover = [p for p in target_dir.iterdir() if p.suffix == ".tmp"]
        assert leftover == []


def test_cache_corruption_warning_is_userwarning_subclass() -> None:
    assert issubclass(cache_mod.CacheCorruptionWarning, UserWarning)
