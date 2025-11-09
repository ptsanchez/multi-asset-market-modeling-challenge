# Multi-Asset Market Modeling Challenge

## Introduction

This project aims to investigate whether short-horizon predictive structure exists in a large, multi-asset dataset covering global equities, futures, FX, commodities, sector ETFs, volatility indices, and major cryptocurrencies. The goal was to:

1. Form a financially motivated hypothesis
2. Build and test models using walk-forward evaluation
3. Identify structural insight or a repeatable predictive edge
4. Demonstrate reasoning, ablations, and awareness of pitfalls

## Initial Hypothesis - Predicting 1-Minute Direction

### Motivation

The simplest question you can ask is also the most fundamental:

> Does an asset's recent price history contain enough information to predict the next 1-minute return direction?

Lagged returns are a minimal assumption and serve as a clean baseline that was opted in for this challenge.

### Baseline Model

A simple logistic regression model attempts to predict the next-minute return direction of a target asset using only the past 5 minutes of returns. This would establish a benchmark and help quantify how much (if any) predictability using strictly local lag based features.
Running experiments on the original baseline implementation produced the following results:

| Asset                         | Average Accuracy |
| ----------------------------- | ---------------- |
| **AMZN (Amazon.com Inc)**     | **87.08%**       |
| **DX (Dollar Index Futures)** | **80.00%**       |
| **USDBRL (FX)**               | **83.12%**       |
| **VIX (Volatility Index)**    | **87.36%**       |
| **ETH (Crypto)**              | **51.46%**       |
| **BTC (Crypto)**              | **53.76%**       |

At first glance, it seemed that the baseline was able to capture a predictive edge with the simple features. However, notice the degreaded performance for only particular assets. BTC and ETH crypto assets have an average accuracy across 5 splits of 50%, essentially a coin flip/random choice between the two directions. 

This was the big indicator to discovering why this was the case: **Crypto trades 24/7.** The very high accuracy in the baseline accuracies was a **structural artifact** from the preprocessing, not true predictive power. Many non-24/7 assets have large overnight gaps.
Using ffill() on missing rows created long flat segments where the model trivially learned that “no movement” predicted “no movement next minute.”

This led to reframing the initial hypothesis:

- The microstructure of assets (i.e., how often they trade) creates an illusion of predictability when using naive resampling.
- To find any real predictive value, a model must first filter for perious of actual market activity.

To validate this diagnostic, a modified/corrected baseline model was implemented, where the averagy accuracies fell down to ~50%, back to random choice:

| Experiment                     | Average Accuracy |
| ----------------------------- | ---------------- |
| **AMZN (artifact present)**     | **~80-87%**       |
| **BTC/ETH (no artifact)**       | **~50-53%**       |
| **AMZN w/ corrected baseline**  | **~50%**       |

This shows that there is no predictive edge in 1-minute direction using strictly lagged returns. This demonstrates a key pitfall, admittedly highlighted in the challenge, where good results can arise from data construction rather than real market structure.

### Regime Model

After proving the simple baseline was flawed, the next logical step was to test if a more complex, multi-asset model could find a signal. The challenge document notes that market behavior is not static and that relationships may change across regimes. This inspired the next hypothesis.

**A simple lag-based model fails because it it 'regime-blind.' A model that knows the current multi-asset markey regime (e.g., calm vs panic) can find a predictive edge that the simple model misses.**

To test this, a new HMM/GMM model was build. It was trained on cross-asset features (volatility and correlations from assets like BTC, GOOG, and VIX) to identify latent market states. 
The inferred regime (state 0, 1, or 2) was then added as a new feature to the corrected baseline logistic regression model to see if knowing the regime provided any predictive lift. 

This new, regime-aware model was tested against the same 1-minute target. Results were a definitive null and showed no improvement over corrected 50% baseline beyond statistical noise:

| Experiment                         | Average Accuracy |
| ----------------------------- | ---------------- |
| **Regime Model (global Features)**     | **50.02%**       |
| **Regime Model (global Features)** | **50.45%**       |

Even a complex, multi-asset regime model failed to find any signal for 1-minute direction. After evidence-based iteration, it was shown that the target variable itself is likely intractable, low-signal problem. The noise simply overwhelsm any potential signal from simple lags and broader market states.

This justified moving to a new modeling target.

### Pivot to New Hypothesis: Forecasting 10-Minute Volatility

Instead of trying to predict *direction*, I aimed to predict something these features could actually measure: **volatility.** 

> While cross-asset regimes based on 1-minute data fail to predict short-term direction, they show significant power in predicting near-term volatility.

This modeling target was changed from 1-minute direction to a 10-minute volatility forecast. This required creating a new baseline.
