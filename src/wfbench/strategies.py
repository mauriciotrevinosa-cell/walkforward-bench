"""Rules under study.

A strategy maps a close-price series to a **position series in [0, 1]**: the
fraction of the allowed size to hold on the *next* bar. The bench applies the
one-bar shift, so a strategy must never look forward itself. Getting this wrong
is the single most common way a backtest produces a result that cannot be
traded, and it is subtle enough that it is worth the bench owning the shift
rather than trusting each rule to remember it.

`buy_and_hold` is included as a strategy so that the machinery can be checked
against a case whose answer is known: run through the same folds, the same costs
and the same comparison, it must come out exactly flat against its own baseline.
Any drift there is a bug in the bench, not a finding about markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

PositionRule = Callable[[pd.Series, Mapping[str, Any]], pd.Series]


@dataclass(frozen=True)
class Strategy:
    """A parameterised rule, plus the grid of settings to sweep over."""

    strategy_id: str
    name: str
    description: str
    build: PositionRule
    param_grid: Mapping[str, Sequence[Any]]

    def configurations(self) -> list[dict[str, Any]]:
        """Every valid combination in the grid, in a stable order.

        Stable ordering matters: a sweep whose configuration order depends on
        set iteration produces a different "best" config between runs on ties,
        and the resulting irreproducibility is very hard to see.
        """
        combos: list[dict[str, Any]] = [{}]
        for key in sorted(self.param_grid):
            combos = [
                {**base, key: value}
                for base in combos
                for value in self.param_grid[key]
            ]
        return [combo for combo in combos if self.is_valid(combo)]

    def is_valid(self, params: Mapping[str, Any]) -> bool:
        """Reject nonsense combinations before they are tested.

        A grid over fast and slow windows contains pairs where fast >= slow.
        Testing them inflates the denominator of the selection-bias report with
        configurations that were never candidates.
        """
        fast, slow = params.get("fast"), params.get("slow")
        if fast is not None and slow is not None and int(fast) >= int(slow):
            return False
        return True


def _sma_crossover(close: pd.Series, params: Mapping[str, Any]) -> pd.Series:
    """Long while the fast average is above the slow one, flat otherwise."""
    fast = close.rolling(int(params["fast"])).mean()
    slow = close.rolling(int(params["slow"])).mean()
    return (fast > slow).astype(float)


def _rsi_reversion(close: pd.Series, params: Mapping[str, Any]) -> pd.Series:
    """Long after the market has fallen, flat after it has risen."""
    window = int(params["window"])
    threshold = float(params["threshold"])

    change = close.diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = (-change.clip(upper=0)).rolling(window).mean()
    # A window with no losses has infinite relative strength; force RSI to 100
    # rather than let a division by zero propagate a NaN into the positions.
    strength = gain / loss.replace(0.0, pd.NA)
    rsi = 100 - 100 / (1 + strength)
    rsi = rsi.fillna(100.0)

    return (rsi < threshold).astype(float)


def _buy_and_hold(close: pd.Series, _params: Mapping[str, Any]) -> pd.Series:
    """Fully invested, always. The control case."""
    return pd.Series(1.0, index=close.index)


REGISTRY: dict[str, Strategy] = {
    "sma_crossover": Strategy(
        strategy_id="sma_crossover",
        name="Moving average crossover",
        description="Long while a fast moving average sits above a slow one.",
        build=_sma_crossover,
        param_grid={
            "fast": (5, 10, 20, 30, 50),
            "slow": (20, 50, 100, 150, 200),
        },
    ),
    "rsi_reversion": Strategy(
        strategy_id="rsi_reversion",
        name="RSI mean reversion",
        description="Long while RSI sits below a threshold.",
        build=_rsi_reversion,
        param_grid={
            "window": (7, 14, 21),
            "threshold": (20, 30, 40, 50),
        },
    ),
    "buy_and_hold": Strategy(
        strategy_id="buy_and_hold",
        name="Buy and hold",
        description="The control. Must come out exactly flat against its own baseline.",
        build=_buy_and_hold,
        param_grid={},
    ),
}


def get_strategy(strategy_id: str) -> Strategy:
    if strategy_id not in REGISTRY:
        raise KeyError(f"unknown strategy {strategy_id!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[strategy_id]
