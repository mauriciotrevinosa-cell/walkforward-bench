"""Most of these tests are about the bench refusing to flatter a strategy."""

import numpy as np
import pandas as pd
import pytest

from wfbench import MIN_FOLDS, evaluate_config, get_strategy, sweep
from wfbench.bench import _majority_probability, _returns_with_costs, _windows


def prices(n=1500, drift=0.0003, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.Series(close, index=pd.date_range("2019-01-01", periods=n, freq="B"))


# --- the machinery itself ----------------------------------------------------


def test_test_windows_never_overlap():
    """Overlapping windows count one good quarter several times."""
    splits = _windows(n_bars=1000, train_bars=252, test_bars=63)

    for (_, _, end), (_, next_start, _) in zip(splits, splits[1:]):
        assert end == next_start, "a bar must be scored in exactly one fold"


def test_training_window_is_anchored_and_grows():
    splits = _windows(n_bars=1000, train_bars=252, test_bars=63)
    train_ends = [train_end for train_end, _, _ in splits]

    assert train_ends == sorted(train_ends)
    assert train_ends[0] == 252


def test_buy_and_hold_comes_out_exactly_flat_against_its_own_baseline():
    """The control case. Any drift here is a bug in the bench, not a finding."""
    report = sweep("buy_and_hold", prices(), cost_bps=0.0)

    assert report.configs_tested == 1
    for fold in report.results[0].folds:
        assert fold.excess == pytest.approx(0.0, abs=1e-9)


def test_the_position_is_shifted_so_a_rule_cannot_trade_on_its_own_signal():
    """A rule that goes long on the bar it detects the move must not be paid for it."""
    close = pd.Series([100.0, 100.0, 110.0, 110.0], index=pd.RangeIndex(4))
    # Perfect foresight: long exactly on the bar that jumps.
    positions = pd.Series([0.0, 0.0, 1.0, 0.0], index=close.index)

    total, _ = _returns_with_costs(close, positions, cost_bps=0.0)

    # The shift means the position taken on bar 2 earns bar 3's return, which is
    # zero. Without the shift this would return 10%.
    assert total == pytest.approx(0.0, abs=1e-9)


def test_costs_are_actually_charged():
    close = pd.Series([100.0] * 10, index=pd.RangeIndex(10))
    flipping = pd.Series([float(i % 2) for i in range(10)], index=close.index)

    free, _ = _returns_with_costs(close, flipping, cost_bps=0.0)
    charged, changes = _returns_with_costs(close, flipping, cost_bps=50.0)

    assert free == pytest.approx(0.0, abs=1e-9)
    assert charged < -0.01, "flat prices plus churn must lose money"
    assert changes > 0


def test_raising_costs_can_only_hurt():
    close = prices(seed=3)
    strategy = get_strategy("sma_crossover")
    params = {"fast": 10, "slow": 50}

    cheap = evaluate_config(strategy, params, close, 252, 63, cost_bps=1.0)
    dear = evaluate_config(strategy, params, close, 252, 63, cost_bps=50.0)

    assert dear.mean_excess < cheap.mean_excess


# --- the guards --------------------------------------------------------------


def test_a_short_series_is_refused_rather_than_scored_on_two_folds():
    short = prices(n=300)
    result = evaluate_config(get_strategy("sma_crossover"), {"fast": 5, "slow": 20},
                             short, 252, 63, cost_bps=5.0)

    assert not result.evaluated
    assert f"below the {MIN_FOLDS} required" in result.excluded_reason


def test_a_rule_that_barely_trades_is_excluded_from_the_verdict():
    """Three trades over six years is not a strategy with a good Sharpe."""
    close = prices(n=1500, drift=0.0008, vol=0.002, seed=7)  # smooth uptrend
    result = evaluate_config(get_strategy("sma_crossover"), {"fast": 50, "slow": 200},
                             close, 252, 63, cost_bps=5.0)

    if result.total_position_changes < 10:
        assert not result.evaluated
        assert "position changes" in result.excluded_reason
        assert result.beats_baseline is False, "an excluded config can never be a winner"


def test_invalid_grid_combinations_never_reach_the_denominator():
    """fast >= slow is not a candidate, and must not inflate configs_tested."""
    strategy = get_strategy("sma_crossover")
    full_grid = 5 * 5
    valid = strategy.configurations()

    assert len(valid) < full_grid
    assert all(c["fast"] < c["slow"] for c in valid)


def test_configuration_order_is_stable_across_calls():
    """Otherwise ties resolve differently between runs and results drift."""
    strategy = get_strategy("sma_crossover")
    assert strategy.configurations() == strategy.configurations()


# --- winning honestly --------------------------------------------------------


def test_one_lucky_fold_does_not_count_as_beating_the_baseline():
    """Positive mean is not enough; it must also win more folds than it loses."""
    close = prices(seed=11)
    result = evaluate_config(get_strategy("sma_crossover"), {"fast": 10, "slow": 50},
                             close, 252, 63, cost_bps=5.0)

    if result.evaluated and result.mean_excess > 0:
        assert result.beats_baseline == (result.folds_won * 2 > len(result.folds))


def test_the_null_probability_is_below_a_half_because_ties_are_not_wins():
    """With an even number of folds, a 50/50 split is not a majority."""
    assert _majority_probability(4) < 0.5
    assert _majority_probability(5) == pytest.approx(0.5)
    assert _majority_probability(0) == 0.0


def test_the_sweep_reports_its_denominator_not_just_a_winner():
    report = sweep("sma_crossover", prices(seed=13), symbol="TEST")

    assert report.configs_tested > 1
    assert report.expected_by_chance > 0
    assert str(report.configs_tested) in report.verdict
    assert "buy-and-hold" in report.verdict


def test_a_sweep_with_no_winners_says_so_without_hedging():
    """The most useful result a research bench produces is usually this one."""
    # A pure random walk with no drift, and costs high enough to bite.
    report = sweep("sma_crossover", prices(drift=0.0, seed=17), cost_bps=25.0)

    if report.configs_beating_baseline == 0:
        assert "Nothing here is worth trading" in report.verdict


def test_a_result_at_or_below_chance_is_never_reported_as_a_finding():
    """Two sub-chance outcomes, two different sentences, neither of them a win."""
    report = sweep("rsi_reversion", prices(seed=19), cost_bps=15.0)
    beat, expected = report.configs_beating_baseline, report.expected_by_chance

    if beat == 0:
        assert "Nothing here is worth trading" in report.verdict
    elif beat < expected * 0.5:
        # Worse than a coin flip is a stronger statement than noise, and
        # collapsing it into "noise" throws the more useful finding away.
        assert "worse than random" in report.verdict
    elif beat <= expected:
        assert "indistinguishable from noise" in report.verdict
    else:
        assert "not yet evidence" in report.verdict


def test_well_below_chance_is_called_worse_than_random_not_noise():
    """Constructed directly: a report where almost nothing cleared the bar."""
    from wfbench.bench import SweepReport

    report = SweepReport(
        strategy_id="x", symbol="X", configs_tested=21, configs_excluded=0,
        configs_beating_baseline=1, expected_by_chance=10.5,
        folds_per_config=27, cost_bps=5.0, best=None,
    )

    assert "worse than random" in report.verdict
    assert "noise" not in report.verdict


def test_even_a_result_above_chance_is_not_called_evidence():
    """Selecting the winner after seeing the results is what the sweep is measuring."""
    report = sweep("sma_crossover", prices(drift=0.0008, seed=23), cost_bps=1.0)

    if report.configs_beating_baseline > report.expected_by_chance:
        assert "not yet evidence" in report.verdict


def test_an_unknown_strategy_is_refused():
    with pytest.raises(KeyError, match="unknown strategy"):
        get_strategy("martingale")
