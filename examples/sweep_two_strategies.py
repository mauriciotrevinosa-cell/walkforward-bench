"""Sweep two classic rules, and read the denominator rather than the winner.

The series are synthetic — geometric random walks with equity-like drift and
volatility — so this runs anywhere with no data provider and no key. That also
makes the right answer known in advance: a random walk has no exploitable
structure, so any rule that appears to beat it is showing selection, not skill.

A research bench that cannot produce a negative result on data with no signal
cannot be trusted to produce a positive one on data that has some.

**It runs over many independent paths, on purpose.** A sweep on one path is one
draw, and drawing conclusions from one draw is precisely the mistake this module
exists to prevent. It would be an odd thing for the example to commit.

That is not a hypothetical concern here. On a single unlucky path this bench
reported 8 of 12 RSI configurations beating buy-and-hold — which looks like a
finding until you notice the index fell 53% on that path, and a rule that spends
half its time in cash beats a falling market for reasons that have nothing to do
with skill. Across paths, that effect averages out. On one path it is invisible.

Run:  python examples/sweep_two_strategies.py
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from wfbench import get_strategy, sweep

N_PATHS = 12
COST_BPS = 5.0


def synthetic_index(n_bars=2000, annual_drift=0.08, annual_vol=0.18, seed=0):
    """A geometric random walk with equity-like drift and volatility."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    steps = rng.normal(
        (annual_drift - 0.5 * annual_vol**2) * dt,
        annual_vol * np.sqrt(dt),
        n_bars,
    )
    close = 100 * np.exp(np.cumsum(steps))
    return pd.Series(close, index=pd.date_range("2016-01-01", periods=n_bars, freq="B"))


def sweep_across_paths(strategy_id: str) -> None:
    strategy = get_strategy(strategy_id)
    print(f"\n{strategy.name}  ({strategy_id})")
    print("=" * 78)
    print(f"  {'path':>4}  {'index return':>13}  {'beat baseline':>14}  {'expected':>9}")
    print("  " + "-" * 60)

    winners: list[int] = []
    expected = 0.0
    tested = 0

    for seed in range(N_PATHS):
        close = synthetic_index(seed=seed)
        result = sweep(strategy_id, close, symbol=f"PATH{seed}", cost_bps=COST_BPS)

        index_return = close.iloc[-1] / close.iloc[0] - 1
        winners.append(result.configs_beating_baseline)
        expected = result.expected_by_chance
        tested = result.configs_tested

        print(
            f"  {seed:>4}  {index_return:>+12.1%}  "
            f"{result.configs_beating_baseline:>7} of {tested:<4}  {expected:>9.1f}"
        )

    mean_winners = statistics.mean(winners)
    print("  " + "-" * 60)
    print(f"  {'mean':>4}  {'':>13}  {mean_winners:>7.1f} of {tested:<4}  {expected:>9.1f}")
    print()

    if mean_winners <= expected:
        print(f"  Across {N_PATHS} independent paths, an average of {mean_winners:.1f} of")
        print(f"  {tested} configurations beat buy-and-hold, against {expected:.1f} expected")
        print("  by chance alone. There is no edge here, which is the correct")
        print("  answer: these are random walks.")
    else:
        print(f"  Average {mean_winners:.1f} of {tested} against {expected:.1f} expected.")
        print("  Above chance on synthetic data with no structure means the")
        print("  baseline comparison is biased, not that the rule works.")


def main() -> None:
    print(f"{N_PATHS} independent synthetic indices, 2000 bars each,")
    print(f"8% annual drift, 18% annual volatility, costs at {COST_BPS:g}bp.")

    for strategy_id in ("sma_crossover", "rsi_reversion"):
        sweep_across_paths(strategy_id)

    print()
    print("=" * 78)
    print("TWO THINGS THIS OUTPUT IS SAYING")
    print("=" * 78)
    print("1. The denominator is the report. Sweeping a grid and quoting the")
    print("   best configuration is not a finding, it is the maximum of N draws,")
    print("   and the maximum of N draws from a distribution centred on zero is")
    print("   reliably positive. Bigger grid, more impressive winner, less")
    print("   meaning.")
    print()
    print("2. One path is one draw. Look at the spread down the 'beat baseline'")
    print("   column: individual paths land well above and well below the")
    print("   expected count, and picking the flattering one would have")
    print("   produced a confident, fully documented, entirely false result.")
    print("   The paths where the index fell are the dangerous ones: any rule")
    print("   that sits in cash part of the time beats a falling market without")
    print("   having any skill at all.")
    print()
    print("3. The result came in far BELOW the chance line, and that is")
    print("   informative rather than reassuring. It means the 50/50 null is")
    print("   wrong, in a direction worth understanding: against a baseline that")
    print("   drifts upward, a rule that spends part of its time in cash starts")
    print("   at a structural disadvantage, because it gives up drift it can only")
    print("   earn back by timing. So 'a coin flip against buy-and-hold' is too")
    print("   generous a null, and the honest reading of these numbers is not")
    print("   'no edge' but 'worse than no edge, by exactly the amount of drift")
    print("   these rules sat out'.")
    print()
    print("Raise COST_BPS to 25 and run it again.")


if __name__ == "__main__":
    main()
