# walkforward-bench

Anchored walk-forward strategy evaluation that **reports its own selection
bias** — so that a sweep produces a denominator, not a winner.

```python
from wfbench import sweep

report = sweep("sma_crossover", close, cost_bps=5.0)
print(report.verdict)
# 1 of 21 configurations beat buy-and-hold, well below the 10.5 expected by
# chance. This is worse than random: the rules gave up baseline drift while out
# of the market and did not time their way back to it.
```

---

## About the scope of this repository

**This is a deliberately reduced version of one module of a larger, private
system.** The full bench is part of **Atlas**, a proprietary quantitative
platform I build and maintain, where it sits in front of a live promotion path:
a rule cannot be traded until it has produced broker-owned evidence through a
separate and stricter gate that this bench deliberately cannot reach.

What was removed on purpose:

- the live promotion path, the broker integration and every execution surface
- the data layer, so this runs on synthetic series with no provider and no key
- the persistence store and the decision audit trail
- the real strategy registry — the two rules here are textbook ones, chosen
  because everyone can already judge them

---

## The problem it exists for

Sweeping a parameter grid and reporting the best configuration is not a finding.
It is the maximum of N draws, and the maximum of N draws from a distribution
centred on zero is reliably positive. The bigger the grid, the more impressive
the winner, and the less it means.

So this bench never reports a winner without its denominator. Every sweep states
how many configurations were tried, how many cleared the bar, and how many would
have cleared it by chance alone.

### What it does to be hard to fool

| | |
|---|---|
| **Out of sample only** | Every number comes from a window that follows the data the parameters were chosen on. |
| **Non-overlapping test windows** | One good quarter cannot be counted in five folds. |
| **Costs on every position change** | A rule that only works at zero cost does not work. |
| **Baseline over the identical windows** | Not the full history. Comparing a strategy's test-period return to buy-and-hold's full-period return compares two different questions. |
| **The shift lives in one place** | The bench applies the one-bar position shift, so no rule can trade on the bar it used to decide. |
| **Winning needs more than a good mean** | A rule must also win more folds than it loses. Losing four windows and being rescued by an enormous fifth is one lucky quarter, and a mean alone cannot tell the two apart. |
| **Rules that barely trade are excluded** | Three trades over six years is not a strategy with a good Sharpe ratio, it is three trades. |

---

## Measured results

`examples/sweep_two_strategies.py` runs both rules across **twelve independent
synthetic indices** — geometric random walks with 8% drift and 18% volatility, so
there is no structure to find and the right answer is known in advance.

```
Moving average crossover
  path   index return   beat baseline   expected
     0        -12.3%        1 of 21         10.5
     1        +21.9%        0 of 21         10.5
     2        -54.0%       11 of 21         10.5
     3       +203.5%        0 of 21         10.5
     4        +93.7%        0 of 21         10.5
     ...
  mean                     1.2 of 21         10.5
```

Three things in that table, in ascending order of how much they matter.

**One path is one draw.** Look at the spread. Path 2 shows 11 of 21
configurations beating the baseline; nine other paths show zero. A sweep run on
one series would have produced a confident, fully documented result, and which
result it produced would have depended entirely on which series.

**Path 2 is the trap, and it is worth understanding exactly.** Its index fell
54%. Any rule that spends part of its time in cash beats a falling market, for
reasons that have nothing whatever to do with skill. That is not a subtle
artefact — it is the single most common way a momentum or reversion backtest
manufactures an edge, and on a single down-trending sample it is invisible.

**The result landed far below the chance line, and that is informative rather
than reassuring.** 1.2 against 10.5 expected is not noise; it is worse than a
coin flip. Which means the 50/50 null is wrong, in a direction worth
understanding: against a baseline that drifts upward, a rule that sits in cash
part of the time starts at a structural disadvantage, because it gives up drift
it can only earn back by timing. The honest reading is not "no edge" but "worse
than no edge, by roughly the drift these rules sat out".

### The result this was built to produce

Run against real price history inside Atlas, the same bench measured **0 of 105
configurations beating buy-and-hold out of sample**. No rule was promoted, and
`sma_crossover` in particular was never allowed near a live path.

That is a negative result and it is the most valuable thing this module has ever
produced. A research bench that cannot return "nothing here" is not a research
bench, it is a machine for generating justifications.

---

## Running it

```bash
pip install -e ".[dev]"
pytest -q
python examples/sweep_two_strategies.py
```

18 tests. The ones worth reading are the traps:

- `test_the_position_is_shifted_so_a_rule_cannot_trade_on_its_own_signal` — gives
  a rule perfect foresight and asserts it earns nothing.
- `test_buy_and_hold_comes_out_exactly_flat_against_its_own_baseline` — the
  control case. Any drift there is a bug in the bench, not a finding about
  markets.
- `test_test_windows_never_overlap` — asserted directly on the split indices.
- `test_invalid_grid_combinations_never_reach_the_denominator` — `fast >= slow`
  pairs are not candidates and must not inflate the selection-bias report.
- `test_well_below_chance_is_called_worse_than_random_not_noise` — because
  collapsing "worse than a coin flip" into "noise" throws away the better finding.

---

## How this repository was built

I want this stated plainly rather than left to be inferred.

**The system this came from is mine.** Atlas is a platform I have been building
and maintaining, and the engineering judgement in it — the architecture, the
decisions about what a component must refuse to do, the discipline this module
demonstrates — is work I did.

**This repository is not that code.** It was written with **Claude Code
(Anthropic)**, working from my Atlas source and my design decisions, to produce
a smaller self-contained version that can be read and judged in twenty minutes
without exposing the private system. Claude wrote most of the code here, wrote
the tests, ran them, and found several real bugs in the process — those are
named in the sections above. I decided what to extract, what to leave out, and
which properties the reduced version had to preserve. The commits carry
`Co-Authored-By: Claude Opus 5`.

**Why say so.** Two reasons. It is true, and I would rather be judged on what I
actually did than on an impression I allowed to stand. And it sets the right
expectation about how I work: I design systems, decide what they must guarantee,
and use the tools available to get them built and verified. I am not claiming to
have typed every line here unaided, and I would rather you know that before we
talk than after.

If you want to test the understanding rather than the authorship, ask me about
any decision documented above. Every one of them is a choice I can defend, and
the reasoning is in the code comments because that is where I wanted it.

---

## License

MIT, and it covers only the reduced code in this repository. The full Atlas
platform this module was extracted from is proprietary and is not licensed here.

## Author

Mauricio Gerardo Trevino Saldana
