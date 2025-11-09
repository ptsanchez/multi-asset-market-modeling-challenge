# Multi-Asset Market Modeling Challenge

### Project Summary

This report details the investigation of a large, multi-asset dataset to find predictive value. The project followed an iterative process:
1.  Began with a hypothesis to predict short-term **directional movement**.
2.  Discovered this was intractable and the initial positive results were a **structural artifact**, a key pitfall.
3.  Pivoted to a new, more robust hypothesis: forecasting near-term **volatility**.
4.  Developed a regime-aware model using a Hidden Markov Model (HMM) that demonstrated a significant, specialized predictive lift over a baseline, validating the final hypothesis.

---

## 1. Hypothesis Investigated

The project investigated two distinct hypotheses in sequence.

### Initial (Failed) Hypothesis: 1-Minute Direction
The initial question was:
> "Does an asset's recent price history (5-minute lagged returns) contain enough information to predict the next 1-minute return direction?"

This hypothesis was **proven to be false**. Initial high-accuracy results (~80-87%) were identified as a structural artifact from naively resampling gapping assets (like equities) versus non-gapping assets (like crypto). After correcting for this data leakage, the model's accuracy fell to ~50% (random chance), showing no predictive edge.

### Final (Successful) Hypothesis: 10-Minute Volatility
The failure of the first approach led to a new hypothesis:
> "While 1-minute direction is too noisy, a model that understands the current **multi-asset market regime** (e.g., 'calm' vs. 'panic') can find a predictive edge in forecasting near-term **volatility**."

This hypothesis was **validated**. The reasoning was that while *direction* is noisy, *volatility* is structural and regime-dependent. A model aware of the broader market state should outperform a simple baseline that only looks at the target asset's own history.

---

## 2. Modeling Approach and Reasoning

### Target and Feature Engineering
* **Initial Target:** 1-minute return direction (binary: +1 or 0).
* **Final Target:** 10-minute volatility, defined as the 10-minute rolling standard deviation of 1-minute returns.
* **Regime Features:** The core of the model is an **HMM/GMM** trained on cross-asset features (e.g., volatility and correlations from assets like BTC, GOOG, and VIX) to identify 3 latent market states. The *inferred regime* (State 0, 1, or 2) was then fed as a new, categorical feature into the final prediction model.

### Model Architecture
1.  **Baseline Model:** A standard logistic regression model trained to predict the volatility target using only the target asset's own recent history (lagged volatility).
2.  **Regime-Aware Model:** The same logistic regression model, but with one additional feature: the current market regime inferred by the HMM.

This design allows for a clean ablation study, where any "lift" can be directly attributed to the regime-awareness.

---

## 3. Validation Method

To respect the time-ordered nature of financial data and avoid lookahead bias, a **walk-forward evaluation** was used.

The methodology is based on `sklearn.model_selection.TimeSeriesSplit`, which creates multiple expanding windows (or "splits"). 5 splits were consistently used across all experiments. For each split, the model is trained on past data and tested on unseen future data. The final accuracy is the average performance across all out-of-sample splits. This method ensures the model is always tested on data it has never seen, simulating a realistic trading environment.

---

## 4. Results and Analysis

### Initial Hypothesis Results
As discussed, after correcting the structural artifact, the model showed no predictive power for 1-minute direction.

| Experiment | Average Accuracy (5 Splits) | Interpretation |
| :--- | :---: | :--- |
| **AMZN (w/ artifact)** | ~80-87% | False positive (data leakage) |
| **BTC/ETH (no artifact)** | ~50-53% | No signal |
| **AMZN (corrected)** | ~50% | No signal |

### Final Hypothesis: Volatility Ablation Study
The table below compares the **Baseline** volatility model against the **Regime-Aware** model. The "Predictive Lift" column is the sole metric of success in regards to Regime-Aware model performance.

| Target Asset | Regime Feature Set | Avg. Baseline Acc. | Avg. Regime Model Acc. | **Predictive Lift** |
| :--- | :--- | :---: | :---: | :---: |
| GOOG | Equity Regime | 69.98% | 76.36% | **+6.38%** |
| AMZN | Equity Regime | 72.66% | 79.49% | **+6.83%** |
| NVDA | Equity Regime | 61.24% | 60.91% | **-0.33%** |
| BTC | *Equity* Regime | 74.95% | 41.15% | **-33.80%** |
| **BTC** | **Crypto Regime** | **74.95%** | **87.29%** | **+12.34%** |

These results provide the core insights:
1.  **Proof of Methodology:** The model is not a fluke. It provides a **+12.34% lift** for BTC when using relevant *Crypto-Focused* features and a **+6.83% lift** for AMZN using *Equity-Focused* features.
2.  **Proof of Specialization:** The model's power comes from specialization. Applying irrelevant *Equity* features to BTC resulted in a catastrophic **-33.80%** performance drop. This demonstrates that "more features" is not better; *relevant* features are.
3.  **Nuance:** The "Equity" model did not generalize to NVDA, suggesting the market structure it captured is specific to certain large-cap tech assets and not the entire sector.

### Structural Insight: Why the BTC Model Worked
The HMM identified three distinct crypto market regimes. The model's +12.34% lift comes from its ability to distinguish **State 2**, a rare but critical "shock" state, from a standard high-volatility state.

| State | Frequency | Interpretation | Avg. Volatility (std) | Mean Return | Avg. Pair Correlation |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **1** | 72.16% | "Calm" | 0.000813 | +0.000005 | 0.28 |
| **0** | 24.41% | "Correlated Vol" | 0.001403 | -0.000012 | 0.27 |
| **2** | 3.43% | "Crypto-Specific Shock" | 0.002952 | +0.000020 | **0.12** |

The baseline model can likely distinguish "Calm" (State 1) from "Correlated Vol" (State 0). The regime model's edge comes from identifying **State 2**: an explosive, high-volatility event where correlations simultaneously break down. This is a non-obvious, multi-asset structure that a simple baseline cannot see, and showcases the power of its possible predictive power.

---

## 5. Assessment and Next Steps

### What Worked
* **The iterative process:** Starting with a simple hypothesis allowed for the crucial discovery of the 24/7 trading artifact. This failure was essential for motivating the pivot.
* **The regime-aware methodology:** The core idea that multi-asset regimes provide predictive power was validated.
* **Feature specialization:** The ablation study proved that predictive value is unlocked by correctly specializing multi-asset features to the target's unique market domain (crypto-features-for-crypto).

### What Did Not Work
* **1-Minute Directional Forecasting:** This target is likely intractable (pure noise) with the chosen features. The initial positive results were a clear example of a "pitfall" from data construction, not a real signal.
* **Universal Models:** The "Equity Regime" did not apply to all tech stocks (failing on NVDA), and it was actively harmful when applied to BTC. This confirms a "one-size-fits-all" model is ineffective.

### Next Steps with Additional Time

Due to school/classes, I was not able to spend as much time as I would have liked to this really fun challenge, so a couple of "future work" possibilities include:

1.  **Expand Regime Ablations:** Test the validated "Crypto Regime" model on the other crypto assets (ETH, SOL, etc.) to see if the +12.34% lift on BTC generalizes across the asset class.
3.  **Model Regime Transitions:** Instead of just *using* the current regime, build a model to *forecast the probability of a regime transition* (e.g., moving from "Calm" to "Crypto-Specific Shock"). This could provide a more powerful, forward-looking signal.
