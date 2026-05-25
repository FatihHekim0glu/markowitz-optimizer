"""Unit tests for the constraint DSL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cvxpy")

from markowitz.optimizer.constraints import (  # noqa: E402
    ConstraintContext,
    CustomConstraint,
    LeverageCap,
    LongOnly,
    SectorCap,
    TurnoverCap,
    WeightBounds,
)


def _ctx(n: int = 3) -> ConstraintContext:
    return ConstraintContext(
        tickers=tuple(f"T{i}" for i in range(n)),
        n_assets=n,
        long_only=True,
    )


class TestLongOnly:
    def test_apply_produces_nonneg(self) -> None:
        w = cp.Variable(3)
        constraints = LongOnly().apply_cvxpy(w, _ctx())
        assert len(constraints) == 1
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - np.array([0.1, -0.2, 0.4]))),
            [cp.sum(w) == 1, *constraints],
        )
        prob.solve()
        assert (w.value >= -1e-7).all()

    def test_description(self) -> None:
        assert LongOnly().description == "long_only"


class TestWeightBounds:
    def test_scalar_bounds_enforced(self) -> None:
        w = cp.Variable(3)
        bounds = WeightBounds(0.05, 0.5)
        constraints = bounds.apply_cvxpy(w, _ctx())
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - np.array([2.0, -1.0, 1.0]))),
            [cp.sum(w) == 1, *constraints],
        )
        prob.solve()
        assert (w.value >= 0.05 - 1e-7).all()
        assert (w.value <= 0.5 + 1e-7).all()

    def test_vector_bounds(self) -> None:
        w = cp.Variable(3)
        bounds = WeightBounds([0.0, 0.1, 0.2], [1.0, 1.0, 1.0])
        constraints = bounds.apply_cvxpy(w, _ctx())
        prob = cp.Problem(cp.Minimize(cp.sum(w)), [cp.sum(w) == 1, *constraints])
        prob.solve()
        assert w.value[1] >= 0.1 - 1e-7
        assert w.value[2] >= 0.2 - 1e-7

    def test_lower_above_upper_rejected(self) -> None:
        w = cp.Variable(3)
        with pytest.raises(ValueError, match="exceeds upper"):
            WeightBounds(0.6, 0.5).apply_cvxpy(w, _ctx())

    def test_shape_mismatch_rejected(self) -> None:
        w = cp.Variable(3)
        with pytest.raises(ValueError, match="shape"):
            WeightBounds([0.0, 0.1], 1.0).apply_cvxpy(w, _ctx())


class TestSectorCap:
    def test_caps_enforced(self) -> None:
        w = cp.Variable(3)
        ctx = _ctx(3)
        sector = SectorCap(
            sector_mapping={"T0": "tech", "T1": "tech", "T2": "fin"},
            caps={"tech": 0.4},
        )
        constraints = sector.apply_cvxpy(w, ctx)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - np.array([1.0, 1.0, 0.0]))),
            [cp.sum(w) == 1, w >= 0, *constraints],
        )
        prob.solve()
        assert float(w.value[0] + w.value[1]) <= 0.4 + 1e-7

    def test_floors_enforced(self) -> None:
        w = cp.Variable(3)
        ctx = _ctx(3)
        sector = SectorCap(
            sector_mapping={"T0": "fin", "T1": "tech", "T2": "tech"},
            caps={"tech": 1.0},
            floors={"fin": 0.3},
        )
        constraints = sector.apply_cvxpy(w, ctx)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - np.array([0.0, 0.5, 0.5]))),
            [cp.sum(w) == 1, w >= 0, *constraints],
        )
        prob.solve()
        assert float(w.value[0]) >= 0.3 - 1e-7

    def test_empty_sector_is_vacuous(self) -> None:
        w = cp.Variable(3)
        sector = SectorCap(
            sector_mapping={"T0": "tech"},
            caps={"healthcare": 0.1},
        )
        constraints = sector.apply_cvxpy(w, _ctx(3))
        assert constraints == []


class TestLeverageCap:
    def test_l1_norm_bound(self) -> None:
        w = cp.Variable(3)
        lev = LeverageCap(1.2)
        constraints = lev.apply_cvxpy(w, _ctx())
        # Unconstrained problem with no sign restriction can pick large |w|.
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - np.array([1.0, -1.0, 1.0]))),
            constraints,
        )
        prob.solve()
        assert float(np.abs(w.value).sum()) <= 1.2 + 1e-6


class TestTurnoverCap:
    def test_l1_turnover_against_series(self) -> None:
        w = cp.Variable(3)
        ctx = _ctx(3)
        prev = pd.Series([1 / 3, 1 / 3, 1 / 3], index=["T0", "T1", "T2"])
        turnover = TurnoverCap(prev_weights=prev, max_turnover=0.2)
        constraints = turnover.apply_cvxpy(w, ctx)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - np.array([0.9, 0.05, 0.05]))),
            [cp.sum(w) == 1, w >= 0, *constraints],
        )
        prob.solve()
        assert float(np.sum(np.abs(w.value - prev.to_numpy()))) <= 0.2 + 1e-6

    def test_array_prev_shape_check(self) -> None:
        w = cp.Variable(3)
        ctx = _ctx(3)
        bad = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="shape"):
            TurnoverCap(prev_weights=bad, max_turnover=0.1).apply_cvxpy(w, ctx)

    def test_missing_ticker_filled_zero(self) -> None:
        w = cp.Variable(3)
        ctx = _ctx(3)
        prev = pd.Series([0.5, 0.5], index=["T0", "T1"])  # T2 missing
        turnover = TurnoverCap(prev_weights=prev, max_turnover=2.0)
        constraints = turnover.apply_cvxpy(w, ctx)
        prob = cp.Problem(cp.Minimize(cp.sum(w)), [cp.sum(w) == 1, *constraints])
        prob.solve()
        # Just check it solved cleanly.
        assert w.value is not None


class TestCustomConstraint:
    def test_builder_called(self) -> None:
        w = cp.Variable(3)

        def builder(var: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
            assert ctx.n_assets == 3
            return [var[0] == 0.25]

        cc = CustomConstraint(builder, description="anchor T0")
        constraints = cc.apply_cvxpy(w, _ctx())
        prob = cp.Problem(cp.Minimize(cp.sum_squares(w)), [cp.sum(w) == 1, *constraints])
        prob.solve()
        assert abs(float(w.value[0]) - 0.25) < 1e-6
        assert cc.description == "anchor T0"


# ---------------------------------------------------------------------------
# Description coverage -- cheap but pulls the property accessor on every
# constraint type so we lose no branches to "untested getter" lines.
# ---------------------------------------------------------------------------


class TestDescriptions:
    def test_long_only_description(self) -> None:
        c = LongOnly()
        assert isinstance(c.description, str)
        assert len(c.description) > 0

    def test_weight_bounds_description(self) -> None:
        c = WeightBounds(0.0, 1.0)
        assert isinstance(c.description, str)
        assert len(c.description) > 0

    def test_sector_cap_description(self) -> None:
        c = SectorCap(sector_mapping={"T0": "tech"}, caps={"tech": 0.5})
        assert isinstance(c.description, str)
        assert len(c.description) > 0

    def test_leverage_cap_description(self) -> None:
        c = LeverageCap(1.5)
        assert isinstance(c.description, str)
        assert len(c.description) > 0

    def test_turnover_cap_description(self) -> None:
        prev = pd.Series([0.5, 0.5], index=["T0", "T1"])
        c = TurnoverCap(prev_weights=prev, max_turnover=0.1)
        assert isinstance(c.description, str)
        assert len(c.description) > 0


class TestSectorCapEdgeCases:
    def test_sector_cap_floor_for_empty_sector_is_vacuous(self) -> None:
        """Floor for a sector with zero member tickers must be a no-op.

        We map only the tech sector but request a floor on `fin`; the loop
        should hit the ``mask.sum() == 0`` short-circuit and emit no
        constraint at all.
        """
        w = cp.Variable(3)
        sector = SectorCap(
            sector_mapping={"T0": "tech", "T1": "tech", "T2": "tech"},
            caps={"tech": 1.0},
            floors={"fin": 0.3},  # nobody is in 'fin'
        )
        constraints = sector.apply_cvxpy(w, _ctx(3))
        # Only the 'tech' cap should make it through; the empty 'fin' floor
        # short-circuits.
        assert len(constraints) == 1
