# RWS Challenge - Multi-Asset Market Modeling Challenge 

## Project Overview

This project focuses on understanding time-varying market structure through:

1. A Baseline Predictive Model: A simple logistic regression model attempts to predict the next-minute return direction of an asset using only the past 5 minutes of returns. This will establish a benchmark and help quantify how much (if any) predictability using strictly local lag based features.

2. Multi-Asset Regime Discovery: Inspired by the challenge guidelines, the ultimate goal of the project is to detect latent market regimes using features derived from multiple asset classes (equities, FX, crypto, and volatility indices). These regimes will then be evaluated to determine whether they offer predictive power or structural insight.

## Baseline Model and Interpretation:

To keep things simple and features limited, the baseline model predicts whether an asset's next one-minute return will be positive or negative using only the previous five minutes of returns. To assess whether simple autoregressive structure exists across different markets, the baseline model was run on several assets over 5 splits respectively:

| Asset                         | Average Accuracy |
| ----------------------------- | ---------------- |
| **AMZN (Amazon.com Inc)**     | **87.08%**       |
| **DX (Dollar Index Futures)** | **80.00%**       |
| **USDBRL (FX)**               | **83.12%**       |
| **VIX (Volatility Index)**    | **87.36%**       |
| **ETH (Crypto)**              | **51.46%**       |
| **BTC (Crypto)**              | **53.76%**       |


### Vix, USDBRL, and DX achieve very high prediective accuracy (80-87%)

These assets exhibit strong serial dependence, meaning their minute-to-minute returns contains predictable structure. This is consistent with how markets behave:

Volatility indices (VIX)
- VIX does not trade like a typical asset.
- It is constructed from S&P 500 options and tends to mobe smoothly due to its calculation methodology.
- Intraday VIX changes changes often show autocorrelation, so lagged returns can explain a large portion of next-minute direction.
- Baseline accuracy ~87%

USDBRL (Brazilian Real)
- Emerging marke currency pairs often exhibit microstructure frictions, slower reaction times, and persistent order flow.
- This leads to predctable short-term movements.
- Baseline accuracy ~83%, showing strong autocorrelation in short-horizon returns.

DX (Dollar Index Futures)
- The U.S. dollar index responds to macro flows and tends to trend smoothly intraday.
- DX often shows momentum-like microstructure, especially during liquid hours.
- The baseline model achieves ~80%, confirming meaningful short-term dependence.

### ETH behaves like BTC: near-random accuracy (~52% accuracy)

Crypto markets operate in a very different microstructure regime:
- Price discovery is fast.
- Arbitrage mechanisms across exchanges eliminate short-lived patterns.
- One-minute returns exhibit very low autocorrelation.

As expected, ETH and BTC achieved ~52%, essentially a coin flip. This reinforces the idea that strictly simple lag-based signals do not capture crypto dynamics.

### Key Insight Across Assets

Some markets exhibit stable short-term autocorrelation, while highly liquid crypto assets do not. This confirms that
- Financial time series are nonstationary and behavior varies dramatically by asset class.
- Lag-based models can appear highly predictive in some markets but fail completely in others
- Any predictive edge detected in one regime or asset may not generalize elsewhere.

The split-by-split results show that the first out-of-sample window indicates some form of apparent predictive performence, where the accuracy was meaningfully higher than just a simple coin toss. However, in all subsequence windows, performance fell back to roughly 50%.

| Split              | Split Accuracy  |
| ------------------ | ---------------- |
| **1**              | **66.58%**       |
| **2**              | **50.89**        |
| **3**              | **50.35%**       |
| **4**              | **50.50%**       |
| **5**              | **50.48%**       |

This pattern might suggest that any short-term predictability present in the early period did not persist into later market conditions. In other words, the relationship between recent returns and next-minute direction is not stable over time. This is consistent with properties of high-frequency markets, where return dynamics are highly nonstationary and any transient momentum or microstructure effects often decay quickly. 

Overall so far, short-term predictability is highly asset-dependent. Some markets contain meaningful lagged structure, while others do not. This heterogeneity supports the motivation for the next phase of the project: a multi-asset volatility regime model that explains when and why predictability exists. 
