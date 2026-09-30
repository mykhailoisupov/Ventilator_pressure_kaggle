# Ventilator Pressure Prediction

AI301 Deep Learning — Assignment #1
Kaggle: [Google Brain — Ventilator Pressure Prediction](https://www.kaggle.com/competitions/ventilator-pressure-prediction)

All code is in `notebook.ipynb`, run top to bottom.

---

## 1. The problem and the metric

A ventilator drives air into an artificial lung through an inlet valve and releases it through an
exhaust valve. Given the two control signals and the lung being ventilated, predict the resulting
airway pressure at every timestep.

| Column | Meaning |
|---|---|
| `breath_id` | one breath, exactly 80 consecutive rows |
| `R` | airway resistance, 3 levels: 5, 20, 50 |
| `C` | lung compliance, 3 levels: 10, 20, 50 |
| `time_step` | seconds since the breath started |
| `u_in` | inlet valve opening, 0–100 |
| `u_out` | exhaust valve, 0 = inhaling, 1 = exhaling |
| `pressure` | target, cmH₂O |

The score is mean absolute error **restricted to the inspiratory phase**:

$$\mathrm{MAE} = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \left| y_i - \hat{y}_i \right|,
\qquad \mathcal{S} = \{\, i : u\_out_i = 0 \,\}$$

Only **38.0%** of rows are in $\mathcal{S}$, identical in train and test. Predictions must still be
submitted for every row; the remaining 62% are simply not read.

**Reference points.** The metric is unbounded and has no natural scale, so two trivial models were
measured first to make later numbers interpretable:

| Model | MAE |
|---|---|
| constant prediction (17.60 cmH₂O) | 7.6192 |
| mean pressure per $(R, C, t)$ lookup table | 6.1648 |

**Weaknesses of the metric.** It is not differentiable at zero, it is scale blind (1 cmH₂O of error
counts the same at 5 and at 60 cmH₂O), and it ignores the discreteness of the target discussed in
§2. As complementary metrics I report RMSE, MAE per $(R,C)$ group, MAE per timestep and the share of
rows within 0.5 cmH₂O.

**An alternative metric.** MAE averaged per breath first, plus a penalty on the worst error inside
each breath:

$$\mathcal{M} = \frac{1}{B}\sum_{b=1}^{B}\left( \mathrm{MAE}_b + \lambda \max_{i \in b} |y_i - \hat{y}_i| \right)$$

For a device driving air into a patient, a model that is right on average but spikes once is more
dangerous than one that is slightly off everywhere, and a flat row-level mean cannot express that.
Averaging per breath first also stops the larger $(R,C)$ groups from dominating the score.

---

## 2. What the EDA established

Five findings, each of which changed a later decision.

**(1) The unit of observation is a breath, not a row.** Every breath is exactly 80 rows and no
`breath_id` appears in both train and test. Splitting rows at random would put timestep $t$ in train
and $t+1$ in validation — nearly the same measurement. The split must be by breath (§3).

**(2) `R` and `C` are categorical.** Three levels each, constant within a breath. Their joint
distribution is *not* uniform — $(50,10)$ holds 18.1% of breaths, $(20,10)$ only 8.1% — but train and
test match to within 0.5 pp. They enter the network as embeddings and LightGBM as declared
categoricals.

**(3) The target lives on a grid.** Only **950 distinct values** exist across 6 million rows, evenly
spaced $h \approx 0.0703$ cmH₂O apart (min gap 0.070297, max 0.070305). This is ADC quantisation from
the pressure sensor. If a prediction falls uniformly inside a grid cell, its expected distance to the
nearest legal value is $h/4 \approx 0.0176$ — so rounding predictions onto the grid is free
improvement *provided* the model error is already below about half a step. This was tested, not
assumed (§6).

**(4) Pressure integrates `u_in`, it does not follow it.** On individual breaths, pressure builds
while the valve is open and decays on its own after `u_out` flips. The correlation table in §4
confirms it numerically: raw `u_in` correlates 0.092 with pressure, its cumulative sum 0.582.

**(5) The two phases are different regimes.** The inspiratory phase is driven; the rest is passive
decay. This is why only $u\_out = 0$ is scored, and why the training loss is masked the same way.

Sampling is *not* perfectly uniform — $\Delta t$ has mean 0.0331 s but a standard deviation of
0.0017 s and a maximum of 0.2511 s — so `time_step` differences carry information and the integral
in §4 must use real $\Delta t$ rather than a step count.

---

## 3. Validation

**Strategy: 5-fold `KFold` over `breath_id`.** Because rows are nested inside breaths, splitting the
list of breath ids *is* a group split, and it is far cheaper than running `GroupKFold` over 6M rows.

**Motivation.** The split mechanism matches the organisers' exactly — train and test share no
breath — so the quantity CV estimates is precisely what the leaderboard measures. No time-based
split is needed, because breaths are independent simulations rather than one continuous recording.
No stratification is needed either: each $(R,C)$ cell holds the same share of every fold to within
1 pp, which was checked rather than assumed.

**Adversarial validation.** Logistic regression on breath-level summary statistics (mean, std, max,
first and sum of `u_in`, breath duration, switch timestep, $R$, $C$), trained to separate train from
test under its own 5-fold CV:

$$\mathrm{AUC} = 0.4953 \pm 0.0023$$

Indistinguishable from chance. The sets are draws from the same distribution, so there is no shift to
correct for and local CV should track the leaderboard.

**Where correlation could still break.** Selection bias from tuning repeatedly against the same five
folds — mitigated by fixing the split once and reusing it for every model. The fold-to-fold standard
deviation for the tree was **0.00174**, which is the yardstick used throughout: any improvement
smaller than that is noise.

---

## 4. Feature engineering

The physical model of a single-compartment lung is

$$P(t) \;=\; \underbrace{R\,Q(t)}_{\text{resistive}} \;+\; \underbrace{\frac{V(t)}{C}}_{\text{elastic}} \;+\; P_0,
\qquad V(t) = \int_0^t Q(s)\,ds$$

where $Q$ is flow and $V$ the volume delivered. We never observe $Q$ — we observe `u_in`, the valve
*opening*. Taking $Q \propto u\_in$ as a first approximation, every feature group below is a term of
this equation or a discretisation of it.

### 4.1 Cumulative features — the elastic term

The volume is an integral, so it is approximated by a running sum that respects the irregular
sampling:

$$v\_est_t = \sum_{k \le t} u\_in_k \, \Delta t_k \;\approx\; \int_0^{t} Q\,ds$$

From it, the two halves of the physical model and their sum:

$$p\_elastic_t = \frac{v\_est_t}{C}, \qquad
p\_resist_t = R \cdot u\_in_t, \qquad
p\_phys_t = p\_resist_t + p\_elastic_t$$

`p_phys` is a **zero-parameter estimate of the target**. Also included: `u_in_cumsum`,
`u_in_cummean` (the cumsum without its mechanical growth in $t$), `u_in_cummax`, and
`u_in_future_sum` $= \sum_k u\_in_k - u\_in\_cumsum_t$, the air still to come.

Measured correlation with pressure on the inspiratory rows:

| feature | \|corr\| |
|---|---|
| `p_elastic` | **0.734** |
| `u_in_cumsum` | 0.582 |
| `v_est` | 0.572 |
| `u_in` (raw) | 0.092 |

Raw `u_in` is the single weakest thing that could be fed to the model; dividing its integral by $C$
is eight times stronger. That is the whole argument for this group in one line.

### 4.2 Smoothed features — the impulse response

The lung is a first-order system. Its response to an input is not the input but a convolution with a
decaying exponential of time constant $\tau = RC$:

$$P(t) \;=\; \int_0^t \frac{1}{\tau}\, e^{-(t-s)/\tau}\, u(s)\, ds$$

Discretising this convolution with $\alpha = \Delta t / \tau$ gives a one-line recursion:

$$\hat{p}_t = (1-\alpha)\,\hat{p}_{t-1} + \alpha\, u_t$$

which is **exactly an exponentially weighted moving average**, with pandas' span parameter related by
$\alpha = 2/(\mathrm{span}+1)$. So `u_in_ewm4`, `u_in_ewm12`, `u_in_ewm30` are not generic smoothing —
they are three discretised impulse responses at three different time constants. Three, because
$\tau = RC$ spans $5\times10 = 50$ to $50\times50 = 2500$ and the correct one is unknown in advance.

Their correlations rank exactly as the physics predicts, longest memory strongest:

$$\text{ewm}_{30}: 0.368 \;>\; \text{ewm}_{12}: 0.305 \;>\; \text{ewm}_{4}: 0.209 \;\gg\; u\_in: 0.092$$

Rolling mean, std, max and min over 3 and 9 steps are included as cruder, finite-window versions of
the same idea.

### 4.3 Lag and difference features — local shape

Sampling is discrete, so derivatives are finite differences:

$$\dot{u}_t \approx \frac{u_t - u_{t-1}}{\Delta t}, \qquad
\ddot{u}_t \approx \frac{u_{t+1} - 2u_t + u_{t-1}}{\Delta t^2}$$

implemented as `u_in_diff_back`, `u_in_diff_fwd` and `u_in_accel`, plus raw lags and leads at
$\pm 1 \ldots 4$ chosen from the measured lag-correlation curve.

**Leads are not leakage.** This is not forecasting: `u_in` and `u_out` are given for all 80 timesteps
of every test breath, and only `pressure` is hidden. The model reconstructs a curve from a fully
observed control signal. The same fact is what makes a bidirectional network legitimate in §5.

### 4.4 Why these products must be built by hand

A decision tree computes a piecewise-constant function over **axis-aligned rectangles**. The level
sets of $v/C$ are rays through the origin in the $(v, C)$ plane, and the level sets of $R \cdot u$ are
hyperbolas — neither is axis-aligned, so approximating them requires many splits and many samples per
region. Supplying $p\_elastic$ and $p\_resist$ directly collapses that approximation into a single
feature. This is a property of the model class, not of the data: **trees can only threshold, never
multiply or divide**.

The network does not have this limitation in principle, but the same features were kept for it so
that the two models see identical information and the comparison in §7 is fair.

### 4.5 Breath-level and categorical features

Twelve features constant within a breath (mean, std, max, sum and first `u_in`, total volume, the
switch timestep, position within the inspiratory phase) give every row the context of the whole
breath — legitimate for the same reason leads are.

`R` and `C` become integer codes `R_cat`, `C_cat` and the 9-level `RC_cat`. Relabelling $5,20,50$ to
$0,1,2$ does **not** by itself remove the ordinal assumption — a linear layer on $0,1,2$ still assumes
equal spacing. What removes it is the embedding lookup in §5 and LightGBM's categorical split on set
membership. With only three observed levels there is nothing to interpolate to, so the model needs to
learn three behaviours, not a function.

**Total: 48 features.**

---

## 5. Models

### 5.1 Tree baseline — LightGBM

Trained on the **scored rows only** ($u\_out = 0$): the metric ignores the rest, the exhale is a
different physical regime, and it cuts training time by roughly a third. Information from the exhale
still reaches the model through the leads and the breath-level aggregates, which are computed over
all 80 rows before filtering.

**Objective comparison**, fold 0, `learning_rate = 0.2`, 96 leaves, 3000 rounds:

| objective | fold-0 MAE | converged |
|---|---|---|
| **Huber** ($\delta = 1$) | **0.4987** | no |
| L2 | 0.5288 | no |
| L1 (MAE) | 0.5521 | no |

Huber wins, and **none of the three converged** — early stopping never fired, all were still
improving at 3000 rounds. The table therefore ranks how fast each objective learns, not which is
ultimately best, and it is reported that way rather than as a final verdict.

The ordering is what the gradients predict. For L1 the derivative is
$\partial_{\hat y} |y - \hat y| = -\operatorname{sign}(y - \hat y) \in \{-1, +1\}$: constant magnitude
regardless of how wrong the model is, so it converges slowly. Huber is quadratic near zero and its
gradient shrinks as the residual does.

### 5.2 Deep model — bidirectional LSTM

One breath is one training example of shape $(80, 45)$, predicting all 80 pressures in a single
forward pass. Continuous features are standardised with train statistics; `R`, `C` and `RC` enter as
learned embeddings (8, 8 and 16 dimensions) broadcast across all 80 timesteps.

**Why a recurrent model.** Pressure depends on $V(t) = \int_0^t Q\,ds$ — an accumulation over the
whole breath so far, not a fixed window. The LSTM cell update is

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

which, with the forget gate $f_t \to 1$, is a running sum: a **learnable leaky integrator** with the
same recursive form as $V_t = V_{t-1} + u_t \Delta t$. The architecture has the shape of the equation.
The tree had to be told the history through hand-built summaries; the LSTM learns which integral it
needs.

**Why bidirectional.** Estimating a state using only the past is *filtering*; using the whole record
is *smoothing*, and a smoother has strictly more information. Since `u_in` is fully observed at test
time, this is a smoothing problem. §7 measures the gain rather than assuming it.

**Custom layer** (assignment requirement) — LayerNorm rebuilt from `nn.Module` and `nn.Parameter`:

$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta,
\qquad \mu,\sigma^2 \text{ over the feature axis}$$

Normalising per timestep across features puts breaths with different `u_in` scales on the same
footing. BatchNorm would pool statistics across different breaths in a batch — the wrong axis for a
sequence. Verified against `nn.LayerNorm` to $10^{-5}$.

**Custom optimizer** (assignment requirement) — AdamW rebuilt on `torch.optim.Optimizer`:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = (1 - \eta\lambda)\,\theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

The bias correction matters because $m_0 = v_0 = 0$ biases early steps toward zero. The decay is
**decoupled**: it multiplies $\theta$ directly rather than being added to $g_t$, so it is not rescaled
by $\sqrt{\hat v_t}$ — otherwise parameters with large gradients would receive less decay than
intended. Verified to converge to the exact optimum on a quadratic.

**Loss.** Trained on masked Huber, scored on masked MAE:

$$\mathcal{L} = \frac{\sum_t m_t \, L_\delta(r_t)}{\sum_t m_t}, \qquad m_t = 1 - u\_out_t,
\qquad L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\[2pt] \delta(|r| - \tfrac12\delta) & |r| > \delta \end{cases}$$

Dividing by $\sum_t m_t$ rather than by 80 matters: the latter would shrink the loss by roughly 3×
and make the effective learning rate 3× too small.

**Regularisation and schedule.** Dropout 0.1, weight decay $10^{-2}$, gradient clipping at norm 5
(LSTMs can produce very large gradients), and OneCycleLR with 10% warmup annealing to ≈0.

| | |
|---|---|
| architecture | 3-layer bidirectional LSTM, hidden 256 |
| parameters | 3,906,139 |
| optimizer | custom AdamW, `max_lr` 3e-3, `weight_decay` 1e-2 |
| batch / epochs | 256 / 60 |

---

## 6. Results

### 6.1 LightGBM, 5 folds

| fold | 0 | 1 | 2 | 3 | 4 | **OOF** |
|---|---|---|---|---|---|---|
| MAE | 0.4684 | 0.4723 | 0.4732 | 0.4700 | 0.4720 | **0.4712** |

Fold-to-fold standard deviation **0.00174**. RMSE 0.7674; 69.9% of rows within 0.5 cmH₂O. The gap
between RMSE and MAE shows a minority of rows carrying large errors.

### 6.2 Grid rounding

| | MAE |
|---|---|
| raw | 0.47121 |
| snapped to the 950-value grid | 0.47072 |
| change | **−0.00049** (fold std 0.00174) |

Inside the noise, so **not used**. The reason is quantitative: the model error of ≈0.47 is about
seven grid steps, so rounding lands on the wrong legal value about as often as the right one. It
would only pay off below roughly $h/2 \approx 0.035$.

### 6.3 Which feature groups actually earned their place

Importance measures the *credit* a feature received; it does not measure *necessity*, because two
correlated features split the credit and both look weak. Each group was therefore removed and the
model retrained on fold 0 (shallower settings, base MAE 0.6016 — only the differences matter):

| group | features | share of gain | Δ MAE when removed |
|---|---|---|---|
| lag / lead | 12 | 13.11% | **+0.0748** |
| breath level | 12 | 9.40% | +0.0589 |
| categorical | 3 | 7.86% | +0.0293 |
| **physics** | 4 | **41.70%** | **+0.0132** |
| cumulative | 5 | 23.31% | +0.0112 |
| smoothed | 8 | 2.71% | +0.0059 |
| raw | 4 | 1.90% | +0.0033 |

Every delta is positive, so nothing was dropped. But **importance and ablation disagree, and the
disagreement is the most informative result in this project.** The physics group carries 41.7% of the
gain from only four features, yet removing it costs just +0.0132 — because

$$p\_elastic = \frac{v\_est}{C}$$

is a deterministic function of the cumulative group, which covers for it. The lag/lead group is the
opposite: only 13% of the gain, but the largest damage when removed, because it is the only source of
local shape and nothing substitutes for it.

### 6.4 Architecture ablation

Same features, same loss, same optimizer, same fold, 25 epochs each — architecture is the only
variable:

| model | parameters | fold-0 MAE |
|---|---|---|
| MLP, per timestep | 119,131 | 0.7868 |
| LSTM, one-way | 1,429,083 | 0.3774 |
| **LSTM, bidirectional** | 3,906,139 | **0.3064** |
| *LightGBM, same fold* | — | *0.4684* |

Every gap is two orders of magnitude larger than the 0.00174 fold noise, so all are real.

The MLP is **worse than LightGBM on identical features**, which settles the question: a per-timestep
network adds nothing over a tree. What matters is memory — adding recurrence cuts the error by more
than half (0.787 → 0.377), and adding the backward pass takes another 0.071 off, confirming the
filtering-versus-smoothing argument empirically.

### 6.5 Deep model

60 epochs, fold 0:

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 59 |
|---|---|---|---|---|---|---|---|
| val MAE | 1.6980 | 0.6228 | 0.4715 | 0.3433 | 0.2767 | 0.2395 | **0.2293** |

**The model did not converge** — validation MAE was still falling at the last epoch (0.2319 → 0.2293).
This is undertraining, not a ceiling, and it is the clearest available improvement (§8).

Gradient flow in the final epoch spans $3.47 \times 10^{-6}$ to $4.15 \times 10^{-3}$ — about three
orders of magnitude, with no tensor flat at zero. Signal reaches the early LSTM layers; clipping and
LayerNorm are doing their job.

### 6.6 Error structure — the same in both models

MAE is small at the start of a breath and grows toward the end of the inspiratory phase, **in both
models**. My initial expectation was the opposite (a cold start where the tree's lag features are
still empty); the data says otherwise, for two reasons:

1. The target is an integral, so an error in the rate keeps accumulating step after step.
2. The spread of pressure itself grows with $t$ — at $t = 0$ every breath sits near the same value —
   so there is simply more to get wrong later.

A gradient-boosted tree and a bidirectional LSTM share no architectural assumptions. **That they fail
in the same place means the cause is the task, not the model.** The LSTM shifts the whole curve
down; it does not change its shape.

### 6.7 Ensemble

A two-stage blend, with the weight fitted on the folds *other* than the one being scored so that it is
never fitted and evaluated on the same rows:

$$\hat{y}^{\,\text{ens}} = w \,\hat{y}^{\,\text{LSTM}} + (1-w)\, \hat{y}^{\,\text{LGBM}}$$

| | OOF MAE |
|---|---|
| LightGBM | 0.4684 |
| LSTM | 0.2293 |
| ensemble, $w = 0.96$ | **0.2282** |

The blend places 96% of the weight on the network and gains only 0.0011 — the tree contributes almost
nothing once the LSTM is present. This number is also **optimistic**: with the network trained on one
fold only there were no other folds to fit the weight on, so it fell back to the same rows. An honest
figure requires all five folds.

### 6.8 Leaderboard

*To be completed after submission. Late submissions return both public and private scores.*

| Model | OOF MAE | Public LB | Private LB |
|---|---|---|---|
| constant prediction | 7.6192 | — | — |
| $(R, C, t)$ lookup table | 6.1648 | — | — |
| LightGBM, Huber, 5 folds | 0.4712 | | |
| LightGBM + grid snapping | 0.4707 | — | — |
| LSTM, 60 epochs, 1 fold | 0.2293 | | |
| ensemble | 0.2282 | | |

---

## 7. What failed or did not help

Documented rather than hidden, since each one produced a usable conclusion.

- **Grid snapping** (−0.00049, inside the fold noise). The EDA hypothesis was sound; the model is
  simply not yet accurate enough for it to bite.
- **L1 objective for LightGBM** (0.5521 vs 0.4987 for Huber) — the constant-magnitude gradient
  predicted this.
- **Per-timestep MLP** (0.7868) — worse than the tree it was meant to replace.
- **The smoothed group** returned the least per feature: 8 features for 2.71% of gain and +0.0059
  when removed. The physics is right but the cumulative features already capture most of it.
- **Low learning rate for LightGBM** — `lr = 0.03` was the worst trial, but only because 3000 rounds
  was not enough for it, not because the value is bad. A truncated comparison, reported as such.

---

## 8. Conclusions and next steps

The ordering of results is consistent throughout: **structure beats capacity**. A tree with 48
carefully derived features reaches 0.4712. The same features fed to a memoryless network do worse
(0.7868). Adding recurrence — an architecture whose update rule matches the integral in the physical
model — more than halves the error, and the backward pass improves it again.

**Is the model adequate?** It is 13× better than the $(R,C,t)$ lookup table and roughly 2× better
than the tree baseline, but the deep model never converged, so the reported 0.2293 is a lower bound on
what this architecture can do.

Ranked next steps:

1. **Train to convergence.** Validation MAE was still falling at epoch 59. Raising to ~200 epochs is
   the largest available gain and costs nothing but time.
2. **All five folds** — fold-averaged test predictions, and an honest ensemble weight.
3. **More capacity** (hidden 384, 4 layers) — but only after (1); capacity cannot help a model that
   was stopped early.
4. **Not more features.** The error grows along the breath in *both* model families, which makes it a
   property of the task rather than missing information. Both models already have the integral; the
   tree from `v_est`, the network from its cell state.

