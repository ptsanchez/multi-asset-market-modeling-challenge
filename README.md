# Multi-Asset Regime Modeling - RWS Quant Challenge

## Project Overview

This project focuses on understanding time-varying market structure through:

1. A Baseline Predictive Model: A simple logistic regression model attempts to predict the next-minute return direction of BTC using only the past 5 minutes of returns. This established a benchmark and helps quantify how much (if any) predictability using strictly local lag based features.

2. Multi-Asset Regime Discovery: Inspired by the challenge guidelines, the ultimate goal of the project is to detect latent market regimes using features derived from multiple asset classes (equities, FX, crypto, and volatility indices). These regimes will then be evaluated to determine whether they offer predictive power or structural insight.

## Baseline Model Interpretation:

Running the current baseline model produces the following output:

{
  'split_accuracies': [0.6658, 0.5089, 0.5035, 0.5050, 0.5047],
  'average_accuracy': 0.5376
}


The baseline model predicts whether BTC’s next one-minute return will be positive or negative using only the previous five minutes of returns.
Because short-horizon returns are extremely noisy, a random model would achieve 50% accuracy.

Across five walk-forward splits, the model’s average accuracy was 53.8%, only slightly above random. This small improvement may lie within statistical noise. Most notably, the first split achieved meaningfully higher accuracy, but performance rapidly fell back to ~50% in subsequent periods.

This pattern suggests that:

- Any local predictability present in the early period did not persist,

- The relationship between recent returns and next-minute direction is not stable,

- BTC’s intraday return dynamics are nonstationary and sensitive to regime shifts.

The baseline model achieved ~87% accuracy for AMZN, far higher than expected for a financial time series. However, this accuracy is not indicative of genuine predictive edge. AMZN's minute-level returns exhibit extremely low volatility and strong short-term autocorrelation, meaning the direction of the next return is often trivially related to recent returns. This type of “predictability” is dominated by microstructure mechanics and is not economically meaningful (i.e., it would not produce tradable profits after transaction costs).

In contrast to BTC, where the baseline accuracy was near random, these AMZN results highlight the importance of evaluating economic significance, not just classification accuracy. This further justifies the move toward multi-asset regime modeling, where the goal is to capture structural or cross-asset dynamics rather than microstructure artifacts.
