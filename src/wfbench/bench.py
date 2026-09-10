"""Anchored walk-forward evaluation, with the selection bias reported.

The purpose is to find out whether a rule has any edge, and to be hard to fool
while doing it. Four things do most of that work:

**Out of sample only.** Every reported number comes from a test window that
follows the data the parameters were chosen on. Test windows do not overlap, so
one good quarter cannot be counted five times.

**Costs are never zero.** They are charged on every change in position. A rule
that only works at zero cost does not work, and the cheapest way to manufacture
an edge is to forget to pay for it.

**A baseline over the identical windows.** Not the full history — the same
windows. Comparing a strategy's test-period return to buy-and-hold's full-period
return is a comparison between two different questions.

**The sweep reports its own selection bias.** Trying 105 configurations and
reporting the best one is not a finding, it is a maximum of 105 draws. The
report states how many were tried and how many beat the baseline, so the reader
can see the denominator instead of being shown a winner.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from .strategies import Strategy, get_strategy

# Roughly one year trained, one quarter tested, on daily bars.
DEFAULT_TRAIN_BARS = 252
DEFAULT_TEST_BARS = 63
MIN_FOLDS = 3
# Below this, a "strategy" is a handful of trades and its statistics are noise.
MIN_POSITION_CHANGES = 10


@dataclass(frozen=True)
class Fold:
    """One out-of-sample window and what happened in it."""

    train_end: str
    test_start: str
    test_end: str
    strategy_return: float
    baseline_return: float
    position_changes: int

    @property
    def excess(self) -> float:
        return self.strategy_return - self.baseline_return

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigResult:
    """One parameter configuration, evaluated across every fold."""

    params: dict[str, Any]
    folds: list[Fold]
    total_position_changes: int
    excluded_reason: str = ""

    @property
    def evaluated(self) -> bool:
        return not self.excluded_reason

    @property
    def mean_excess(self) -> float:
        return sum(f.excess for f in self.folds) / len(self.folds) if self.folds else 0.0

    @property
    def folds_won(self) -> int:
        return sum(1 for f in self.folds if f.excess > 0)

    @property
    def beats_baseline(self) -> bool:
        """Positive on average AND in more folds than not.

        Requiring both is deliberate. A strategy that loses in four windows out
        of five and is rescued by one enormous fifth has not shown an edge, it
        has shown one lucky quarter, and a mean alone cannot tell the two apart.
        """
        return self.evaluated and self.mean_excess > 0 and self.folds_won * 2 > len(self.folds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "mean_excess_pct": round(self.mean_excess * 100, 4),
            "folds_won": self.folds_won,
            "folds_total": len(self.folds),
            "position_changes": self.total_position_changes,
            "beats_baseline": self.beats_baseline,
            "excluded_reason": self.excluded_reason,
        }


@dataclass
class SweepReport:
    """The result of testing a whole grid, with the denominator kept in view."""

    strategy_id: str
    symbol: str
    configs_tested: int
    configs_excluded: int
    configs_beating_baseline: int
    expected_by_chance: float
    folds_per_config: int
    cost_bps: float
    best: ConfigResult | None
    results: list[ConfigResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """A sentence that can be quoted without misleading anyone."""
        if self.configs_tested == 0:
            return "No configuration had enough data to evaluate."

        beat = self.configs_beating_baseline
        expected = self.expected_by_chance

        if beat == 0:
            return (
                f"0 of {self.configs_tested} configurations beat buy-and-hold "
                f"out of sample. Nothing here is worth trading."
            )
        if beat < expected * 0.5:
            # Worse than a coin flip is not the same as noise, and reporting it
            # as noise loses the more useful finding. Against a baseline that
            # drifts upward, a rule that spends part of its time in cash gives up
            # that drift and can only earn it back by timing. Landing well below
            # chance says it did not, systematically.
            return (
                f"{beat} of {self.configs_tested} configurations beat buy-and-hold, "
                f"well below the {expected:.1f} expected by chance. This is worse "
                f"than random: the rules gave up baseline drift while out of the "
                f"market and did not time their way back to it."
            )
        if beat <= expected:
            return (
                f"{beat} of {self.configs_tested} configurations beat buy-and-hold, "
                f"at or below the {expected:.1f} expected by chance alone. "
                f"This is indistinguishable from noise."
            )
        return (
            f"{beat} of {self.configs_tested} configurations beat buy-and-hold, "
            f"against {expected:.1f} expected by chance. Worth a second look, "
            f"and not yet evidence: the winners were selected after seeing these results."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "configs_tested": self.configs_tested,
            "configs_excluded": self.configs_excluded,
            "configs_beating_baseline": self.configs_beating_baseline,
            "expected_by_chance": round(self.expected_by_chance, 2),
            "folds_per_config": self.folds_per_config,
            "cost_bps": self.cost_bps,
            "verdict": self.verdict,
            "best": self.best.to_dict() if self.best else None,
            "notes": self.notes,
        }


def _windows(n_bars: int, train_bars: int, test_bars: int) -> list[tuple[int, int, int]]:
    """Anchored walk-forward splits as (train_end, test_start, test_end) indices.

    Anchored means the training window always starts at the beginning and grows;
    only the test window slides. Test windows never overlap, so no bar is scored
    twice and a single good quarter cannot be counted in several folds.
    """
    splits: list[tuple[int, int, int]] = []
    train_end = train_bars
    while train_end + test_bars <= n_bars:
        splits.append((train_end, train_end, train_end + test_bars))
        train_end += test_bars
    return splits


def _returns_with_costs(
    close: pd.Series,
    positions: pd.Series,
    cost_bps: float,
) -> tuple[float, int]:
    """Total return of a position series over `close`, net of trading costs.

    The position is shifted one bar forward here, in one place, so a rule cannot
    accidentally trade on a bar it used to make the decision.
    """
    held = positions.shift(1).fillna(0.0)
    bar_returns = close.pct_change().fillna(0.0)

    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * (cost_bps / 10_000.0)

    net = held * bar_returns - costs
    total = float((1.0 + net).prod() - 1.0)
    changes = int((held.diff().fillna(held).abs() > 1e-9).sum())
    return total, changes


def evaluate_config(
    strategy: Strategy,
    params: Mapping[str, Any],
    close: pd.Series,
    train_bars: int,
    test_bars: int,
    cost_bps: float,
) -> ConfigResult:
    """Run one configuration through every fold."""
    splits = _windows(len(close), train_bars, test_bars)
    if len(splits) < MIN_FOLDS:
        return ConfigResult(
            params=dict(params),
            folds=[],
            total_position_changes=0,
            excluded_reason=f"{len(splits)} folds available, below the {MIN_FOLDS} required",
        )

    positions = strategy.build(close, params)
    folds: list[Fold] = []
    total_changes = 0

    for train_end, test_start, test_end in splits:
        window_close = close.iloc[test_start:test_end]
        # The positions are computed on the full series so that indicators are
        # warm at the start of each test window, then sliced. The rule still
        # only ever sees past bars, because every indicator here is backward
        # looking and the shift is applied inside _returns_with_costs.
        window_positions = positions.iloc[test_start:test_end]

        strategy_return, changes = _returns_with_costs(window_close, window_positions, cost_bps)
        baseline_return = float(window_close.iloc[-1] / window_close.iloc[0] - 1.0)
        total_changes += changes

        folds.append(
            Fold(
                train_end=str(close.index[train_end - 1]),
                test_start=str(close.index[test_start]),
                test_end=str(close.index[test_end - 1]),
                strategy_return=strategy_return,
                baseline_return=baseline_return,
                position_changes=changes,
            )
        )

    result = ConfigResult(params=dict(params), folds=folds, total_position_changes=total_changes)

    if total_changes < MIN_POSITION_CHANGES and strategy.strategy_id != "buy_and_hold":
        result.excluded_reason = (
            f"{total_changes} position changes across all folds, below the "
            f"{MIN_POSITION_CHANGES} needed for the statistics to mean anything"
        )
    return result


def sweep(
    strategy_id: str,
    close: pd.Series,
    symbol: str = "UNKNOWN",
    train_bars: int = DEFAULT_TRAIN_BARS,
    test_bars: int = DEFAULT_TEST_BARS,
    cost_bps: float = 5.0,
) -> SweepReport:
    """Test every configuration in a strategy's grid, and report the denominator.

    Args:
        cost_bps: Round-trip cost in basis points of the position change. The
            default of 5bp is deliberately optimistic for a retail account; the
            point of a parameter is that a reader can raise it and watch results
            they were shown evaporate.
    """
    strategy = get_strategy(strategy_id)
    configurations = strategy.configurations() or [{}]

    results = [
        evaluate_config(strategy, params, close, train_bars, test_bars, cost_bps)
        for params in configurations
    ]

    evaluated = [r for r in results if r.evaluated]
    excluded = [r for r in results if not r.evaluated]
    winners = [r for r in evaluated if r.beats_baseline]

    folds_per_config = len(evaluated[0].folds) if evaluated else 0

    # Under a null of no skill, a configuration is a coin flip against the
    # baseline in each fold, and "beats" requires winning more folds than not.
    # That is the probability of a majority of `folds_per_config` fair flips.
    p_null = _majority_probability(folds_per_config)
    expected_by_chance = len(evaluated) * p_null

    notes = [
        f"costs charged at {cost_bps:g}bp per unit of position change",
        f"baseline is buy-and-hold over the identical test windows, not the full history",
    ]
    if excluded:
        notes.append(
            f"{len(excluded)} configuration(s) excluded before scoring, mostly for trading too rarely"
        )
    if p_null:
        notes.append(
            f"under a no-skill null, a configuration wins a majority of "
            f"{folds_per_config} folds with probability {p_null:.2f}"
        )

    best = max(evaluated, key=lambda r: r.mean_excess) if evaluated else None

    return SweepReport(
        strategy_id=strategy_id,
        symbol=symbol,
        configs_tested=len(evaluated),
        configs_excluded=len(excluded),
        configs_beating_baseline=len(winners),
        expected_by_chance=expected_by_chance,
        folds_per_config=folds_per_config,
        cost_bps=cost_bps,
        best=best,
        results=results,
        notes=notes,
    )


def _majority_probability(n_folds: int) -> float:
    """P(more than half of n fair coin flips come up heads).

    This is the honest null for "beat the baseline in most folds", and it is
    noticeably *less* than 0.5 for even n, because ties are not wins.
    """
    if n_folds <= 0:
        return 0.0
    needed = n_folds // 2 + 1
    return sum(math.comb(n_folds, k) for k in range(needed, n_folds + 1)) / 2**n_folds
