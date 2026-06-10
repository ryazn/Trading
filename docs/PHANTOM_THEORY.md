# PHANTOM Theory — Cross-Market Liquidity-Sweep Reversals on ES

**Purpose of this document.** This is a context document written for an LLM (or a future
analysis session) so that it fully understands the market-structure pattern being traded,
*why* it exists mechanically, what the academic literature says about each phase of it, and —
critically — why every attempt to encode it as a stack of hard boolean filters has failed,
and what the mathematically correct detection architecture is. Treat everything in here as
the ground-truth specification of the theory. When asked to build detectors, backtests, or
indicators for this pattern, conform to the detection principles in §6–§8.

---

## 1. The pattern in one paragraph

A reference market (the "relative-value leg": NQ, XLF/XLK ratio, bonds, breadth — anything
that normally co-moves with ES) **diverges** from ES. This *arms* the setup. ES then makes a
directional push toward an obvious prior extreme (swing high/low) — interpreted as
**seeking liquidity**, i.e. the stop/order cluster resting beyond that level. Price
penetrates the level (the **sweep**), the penetration *fails to hold* (price closes back
through the level), and the market reverses — frequently marking a major swing point of the
session. The terminal reversal shape is highly recognizable to a trained eye, occurs at
"virtually every major swing point within a day's trading", and is **fractal**: the same
anatomy appears on 1-minute charts, 5-minute charts, hourly charts.

The three phases:

| Phase | Name | Observable | Role |
|-------|------|-----------|------|
| A | Divergence (arming) | ES decouples from the relative-value leg | Context / prior — raises the base rate that the ES move is flow-driven, not information-driven |
| B | Liquidity seek | Accelerating push toward a prior swing extreme | The cascade — stop-loss orders trigger sequentially and feed the move |
| C | Terminal sweep & rejection | Wick beyond the level, close back inside, reversal | The trigger — the entry pattern itself |

---

## 2. Why this is not pareidolia: the mechanism, phase by phase

Each phase corresponds to a documented microstructure phenomenon. The pattern is the
*composition* of three effects that the academic literature treats separately.

### Phase B/C mechanics — stop clusters and price cascades (Osler 2005)

Carol Osler's analysis of dealer order books ("Stop-Loss Orders and Price Cascades in
Currency Markets", *Journal of International Money and Finance* 24(2), 2005) established
three empirical facts:

1. Stop-loss orders **cluster** at predictable locations — round numbers and just beyond
   recent extremes (exactly the swing highs/lows the pattern keys off).
2. When price reaches a stop cluster, the resulting move is **unusually rapid and
   self-reinforcing** (positive-feedback trading): triggered stops are market orders that
   push price further into the next stops. This *is* the "liquidity seek" acceleration.
3. These cascades help explain the fat tails of intraday return distributions.

So "the market seeks liquidity" has a precise, non-mystical reading: *once price gets close
enough to a stop cluster, cascade dynamics make reaching and consuming it the
locally-most-likely path*. No conspiratorial "they" is required (though large traders do
deliberately push into clusters when the cost of triggering them is less than the value of
the liquidity released — that behaviour is rational and documented in the stop-hunting
literature).

### Why the reversal after the sweep — order-flow imbalance and depth (Cont–Kukanov–Stoikov 2014)

Cont, Kukanov & Stoikov ("The Price Impact of Order Book Events", *Journal of Financial
Econometrics*, 2014) showed that over short horizons price change is essentially **linear in
order-flow imbalance, with slope inversely proportional to depth**. The sweep is a moment of
maximal one-sided imbalance into minimal depth (the cluster's stops eat the book). Two things
then happen simultaneously:

- The aggressive flow **exhausts** — stops are a finite reservoir. Once consumed, the flow
  that was driving the move literally ceases to exist.
- Passive liquidity **refills against** the move — the stop flow is exactly the counterparty
  volume that large passive players need to build a position in the opposite direction
  without impact. This is why the wick fails: their bids/offers absorb the cascade.

Imbalance flips sign while depth is restored on one side → by the CKS price-impact relation,
price snaps back. The close-back-inside-the-level candle is the visible footprint of that
absorption. (This is also exactly the Wyckoff "spring"/"upthrust", a century-old discretionary
description of the same mechanics.)

### Why it's fractal — self-exciting (Hawkes) order flow

Order arrivals in futures markets are well modelled by **Hawkes processes** (self-exciting
point processes): each market order raises the short-term intensity of further same-side
orders, with power-law-ish decay. Self-excitation has no preferred timescale, which is why
cascade → exhaustion → reversal anatomy reproduces at every resolution. The 1-minute pattern
and the hourly pattern are the same process viewed at different zoom levels. This matters
practically: detector logic should be expressed in **scale-free units** (ATR multiples,
fractions of recent range, event counts) rather than fixed ticks/bars wherever possible.

### Phase A mechanics — why the divergence arms it (price discovery / index arbitrage)

Hasbrouck ("Intraday Price Formation in U.S. Equity Index Markets", *Journal of Finance*,
2003) showed that **price discovery for the S&P complex happens overwhelmingly in the E-mini
(ES)**; SPY, the cash basket, NQ, sector ETFs follow via index/statistical arbitrage. The
consequence for this pattern:

- **Information-driven** moves propagate to the whole complex almost instantly (arbitrage
  enforces co-movement). ES and the reference leg move *together*.
- **Flow-driven** moves (a local stop cascade, a large ES-only order being worked) are
  idiosyncratic to ES — the reference leg does *not* confirm, and a divergence opens up.

So the divergence is a **classifier between information and liquidity**: when ES pushes to a
new extreme *without* the correlated leg confirming, the push is much more likely to be a
liquidity event — i.e. exactly the kind of move that exhausts and reverses when the cluster
is consumed. That is why the divergence "arms" the signal rather than being the signal.

**Answer to the standing question ("can the underlying be ignored?"):** the sweep/rejection
mechanics of Phase C are self-contained in ES — the entry trigger does not need the second
market. But the divergence is the cheapest available estimate of P(move is flow-driven), and
it raises the win rate of otherwise-identical triggers. The correct treatment is therefore:
**never use it as a hard gate, always use it as a weighted score component** (§6). Then the
data, not an opinion, decides how much it matters. Note also the repo's own prior finding
(`QUANT_ANALYSIS.md`): *bar-direction-estimated CVD is statistically noise on 1-min bars* —
the divergence leg should be the relative-value spread (§7), an actual second price series,
not estimated CVD.

### The base-rate asymmetry (the single most important epistemic point)

The observation "this pattern appears at virtually every major swing point" is almost
certainly true **and** is compatible with most sweeps not producing major reversals:

> P(sweep pattern | major intraday reversal) ≈ high — what years of screen time verified.
>
> P(major reversal | sweep pattern) — much lower, because sweeps are far more numerous
> than major reversals.

Major reversal points *are* liquidity events — the day's extremes form precisely where the
largest clusters got consumed — so the conditional in the first direction is nearly
tautological. The entire engineering problem is raising the second conditional: ranking
sweeps by the likelihood they are *terminal*. That is a likelihood-ratio problem, and it is
why the filter-stack approach (§5) was doomed from the start.

---

## 3. Formal definition of the pattern

Let `L` be a swing level (pivot high `H*` for shorts; symmetric for longs), formed at time
`t₀`. The candidate event at time `t > t₀` is:

```
SWEEP(L):     high_t > H* + δ            (penetration, δ ≥ 0 small)
REJECT(L):    close_t < H*               (failure to hold)
```

This minimal two-condition core is the **recall set** — it must catch every pattern instance
a human would mark. Everything else that was previously a filter (level age, penetration
bounds, divergence, virginity, bar size, cooldown) is **evidence**, not definition.

Invariants of the pattern (what is always true):
- A pre-existing, visible reference extreme exists (the cluster location).
- Price approaches it directionally (Phase B has net drift toward the level).
- Penetration occurs and **fails within a small number of bars** (close back through).
- Post-event, the prior drift is broken (changepoint in drift sign).

Variants (what the pattern is *invariant to* — and why no two instances look identical):
- **Time-warping**: the approach can take 10 bars or 60; the rejection can take 1 bar or 4.
- **Amplitude scaling**: sweep depth scales with prevailing volatility.
- **Local noise**: micro-pullbacks during the approach, double-taps of the level.

In ML terms the pattern is an **equivalence class under time-warping and amplitude scaling**,
which is precisely the invariance class that Dynamic Time Warping (DTW) and shapelet methods
were invented for, and precisely the class that fixed-window, fixed-threshold boolean rules
*cannot* represent. This is the mathematical statement of "it's not EXACTLY the same each
time, so it becomes hard to quantify".

---

## 4. What the quant literature offers (toolbox)

Ranked by practical relevance to this exact problem:

1. **Meta-labeling + triple-barrier labeling** — López de Prado, *Advances in Financial
   Machine Learning* (2018). *The* canonical solution to "I have a base signal with good
   recall and poor precision, and hand-tuned filters overfit." Fire the minimal-core signal
   (§3), label each firing by outcome (triple-barrier: target/stop/timeout), then train a
   small secondary model that predicts P(win) per signal from a feature vector. The primary
   signal supplies recall; the meta-model supplies precision. Use purged walk-forward CV.
2. **Soft evidence aggregation (naive-Bayes / logistic scoring)** — the no-ML-infrastructure
   version of the same idea, implementable directly in Pine (§6).
3. **Matrix profile / motif discovery** — Yeh, Keogh et al. (UCR Matrix Profile; STUMPY
   library). Unsupervised, essentially parameter-free (one window length) discovery of
   recurring subsequence motifs. Run on years of 1-min ES windows centred on swing extremes:
   if the terminal pattern is real, it falls out as a top motif *without ever being
   specified*. This is the honest way to validate "it can't be chance" and to extract a
   data-derived template.
4. **DTW / shapelet matching** — subsequence DTW against a small library of canonical
   instances (hand-picked "perfect" examples from chart history) gives a continuous
   similarity score that tolerates warping; shapelet learning extracts the maximally
   discriminative subpattern automatically.
5. **Cointegration / rolling-spread z-score** — the correct formalization of the
   relative-value divergence: z-score each leg over a rolling window, divergence
   `d = z_ES − z_ref` (or a residual from a rolling beta regression). Replaces estimated CVD.
6. **Hawkes-process intensity** — a volume/trade-intensity burst-and-exhaustion measure is a
   cheap proxy for cascade exhaustion: signals where the sweep bar's volume spike is the
   *climax* of a building sequence score higher than isolated pokes.
7. **Changepoint detection (CUSUM / Bayesian online changepoint)** — frames "the reversal"
   as a detected change in drift sign after the sweep rather than as a fixed candle shape;
   useful for confirmation/exit logic and for objective labeling of "did it reverse".
8. **Scale-free swing definition (ε-drawdown, Johansen–Sornette)** — define pivots as moves
   ≥ ε·volatility from an extremum instead of fixed `pivot_left/right` bar counts; makes the
   level inventory itself fractal-consistent.

Key references: Osler (2005) [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=920687),
[NY Fed staff report version](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr150.pdf);
Cont–Kukanov–Stoikov [arXiv:1011.6402](https://arxiv.org/abs/1011.6402);
Hasbrouck (2003) [NYU working paper](https://archive.nyu.edu/bitstream/2451/27374/2/FIN-00-046.pdf);
[UCR Matrix Profile page](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html);
matrix profiles in finance [MDPI](https://www.mdpi.com/2673-4591/5/1/45);
Hawkes order-flow modelling [arXiv:2408.03594](https://arxiv.org/html/2408.03594v1),
[state-dependent Hawkes LOB](https://arxiv.org/pdf/1809.08060);
subsequence-DTW charting [Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417417307376);
shapelet directional forecasting [arXiv:2509.15040](https://arxiv.org/html/2509.15040v1).

---

## 5. Why the filter stack keeps failing (the geometry of overfitting)

Every PHANTOM version to date works like this: a candidate must pass `sweep_ticks ≥ a` AND
`penetration ≤ b` AND `age ≥ c` AND `age ≤ d` AND `divergence ≥ e` AND `bar_size ≤ f` AND
`virgin` AND `cooldown` … Each condition is an axis-aligned cut in feature space; the AND of
N cuts is a **hyperrectangle**. But the good signals do not live in a hyperrectangle — they
live on a manifold where features **trade off against each other**:

- A deeper-than-usual sweep is fine if the rejection close is unusually strong.
- A younger level is fine if the divergence is large and the approach was a clean cascade.
- A slightly oversized bar is fine if it closed pinned to its low.

A hard conjunction cannot represent trade-offs. A candidate that fails *one* gate by one tick
is discarded even if every other dimension is exceptional — which is exactly the reported
failure mode: *"misses others that look clear to me on screen."* The trained eye is not
computing a conjunction; it is computing something like a weighted sum of evidence. Each
added/tightened gate to kill one bad signal also slices off a corner of the manifold
containing good ones — that is the overfit/miss treadmill, and no amount of threshold tuning
exits it, because the model class itself (hyperrectangles) cannot express the target concept.

Secondary problems with the gate architecture:
- **Cliff edges**: infinitesimal input changes flip the output (non-robust, repaint-adjacent).
- **Parameter count**: N gates ≈ N free parameters with interaction effects → tiny effective
  sample per region → overfitting is structurally guaranteed.
- **No diagnosability**: a missed signal leaves no trace of *which* gate killed it or by how
  much.

---

## 6. The correct detection architecture

```
                ┌────────────────────────────────────────────┐
 candidate ───▶ │ MINIMAL CORE (recall layer)                │──▶ rejected (not the pattern)
 (every bar)    │   sweep(L) AND reject-close(L)             │
                └────────────────┬───────────────────────────┘
                                 ▼
                ┌────────────────────────────────────────────┐
                │ EVIDENCE LAYER (precision layer)           │
                │  s₁ sweep-depth shape      (0..1)          │
                │  s₂ rejection strength     (0..1)          │
                │  s₃ level age / maturity   (0..1)          │
                │  s₄ RV divergence          (0..1)          │
                │  s₅ approach quality       (0..1)          │
                │  s₆ volume climax          (0..1)          │
                │  s₇ level freshness        (0..1, soft)    │
                └────────────────┬───────────────────────────┘
                                 ▼
                   score = Σ wᵢ sᵢ / Σ wᵢ    — ONE threshold θ
                                 ▼
              score ≥ θ        → signal
              θ_near ≤ score < θ → "ghost" marker (near-miss, visible for diagnosis)
```

Principles (binding, for any future implementation):

1. **The core stays minimal.** Sweep + rejection close. Nothing else may veto a candidate.
2. **Every former filter becomes a bounded continuous score**, built from smooth ramps or
   logistics of *scale-free* measurements (ATR multiples, range fractions), never raw ticks.
3. **One global threshold** on the aggregate replaces N cliff edges. Sensitivity tuning is
   one knob, and it trades recall against precision *along the manifold* instead of
   chopping corners off it.
4. **Near-misses must be rendered** (ghost markers with score breakdown). When a "clear"
   pattern is missed, the breakdown shows which component under-scored and by how much —
   converting "tune blind, re-test, repeat" into direct diagnosis. This single feature is
   what breaks the overfitting treadmill in practice.
5. **Weights are fit offline, not argued about.** Export per-candidate feature vectors +
   outcomes (triple-barrier), fit a logistic regression (small, interpretable, hard to
   overfit; ~8 parameters vs. thousands of backtests), paste the weights back into the
   indicator. Re-fit on a walk-forward schedule.
6. **Human labels are first-class data.** Marking 200–500 historical candidates as
   good/bad (bar replay) gives a supervised target that captures the trained eye directly —
   meta-label against *that* as well as against trade outcome, and compare.

## 7. The relative-value (divergence) leg — specification

Replace estimated CVD (proven noise at 1-min, see `QUANT_ANALYSIS.md`) with a price-based
spread against a user-chosen reference `R` (NQ, XLF, XLK/XLU, ZN…):

```
z_ES  = (ES  − SMA_n(ES))  / σ_n(ES)         rolling z-score, n ≈ 60–240 bars
z_R   = (R   − SMA_n(R))   / σ_n(R)
d     = z_ES − z_R                            divergence spread
s₄(bear) = logistic(d / d₀)                   ES rich vs ref → bear evidence
s₄(bull) = logistic(−d / d₀)
```

This is a poor-man's cointegration residual: robust, parameter-light, works with any
reasonably correlated reference, and directly encodes "ES extended but the underlying did
not confirm". An upgraded version uses a rolling-beta regression residual; the z-difference
version is preferred first for its lower parameter count. The divergence enters **only as
score component s₄** — if it carries real information, fitted weights will say so; if not,
the signal degrades gracefully to pure-ES sweep logic instead of being held hostage.

## 8. Research roadmap (ordered)

1. **Build the candidate exporter**: minimal-core detector over historical 1-min ES, one row
   per candidate with all sᵢ raw measurements + triple-barrier outcome. (The Python
   backtester in this repo is the natural host.)
2. **Label a few hundred candidates by eye** (bar replay): `good / bad / unclear`.
3. **Fit logistic weights** two ways — against trade outcome, against eye labels — and
   compare. Disagreement between the two is itself informative (the eye may encode context
   the features miss → add features; or the eye may be wrong → trust outcomes).
4. **Run matrix-profile motif discovery** on windows around session swing points as
   independent validation that the terminal motif is real and recurrent, and to extract a
   canonical template for a DTW similarity feature (becomes s₈).
5. **Walk-forward validation** with purged splits; judge by deflated Sharpe / PF on
   *held-out* periods only. Tuning anything against the full sample is forbidden.
6. Only after 1–5: position sizing, session weighting, regime conditioning.

## 9. Glossary (for parsing user shorthand)

- **The pattern / PHANTOM**: the three-phase divergence→seek→sweep-reject structure above.
- **Liquidity sweep / sweep / raid / stop run**: penetration of a prior extreme that
  triggers the stop cluster beyond it.
- **Armed**: Phase-A divergence is active; sweeps occurring while armed are higher-grade.
- **Underlying / relative-value market / RV leg**: the correlated reference series (§7).
- **Virgin level**: a swing extreme not yet penetrated since formation. In v7+, virginity is
  a *soft score*, not a gate.
- **Golden**: legacy term — a signal whose structural target is far enough away to be worth
  taking; in the scoring architecture this folds into trade-management, not detection.
- **Ghost / near-miss**: candidate scoring just below threshold; rendered for diagnosis.
- **Burned level**: previously-touched level (reduced, not zero, evidence value).

---

*Companion implementation: `pinescript/phantom_score.pine` (PHANTOM SCORE v7.0) — the
soft-scoring TradingView indicator implementing §6–§7 with near-miss ghosts and
per-component score breakdown labels.*
