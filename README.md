# RWS Challenge - Multi-Asset Market Modeling Challenge 

## Overview

The objective of this project is to investigate relationships within a large multi-asset dataset and develop a model that demonstrates predictive value or structural insight.
This repository contains all code for data processing, feature engineering, and modeling, as well as the final report and analysis located. 

Report available [here](REPORT.md). 

The project's core logic is containerized in a modular structure, with main.py serving as the primary entry point for running experiments.

## Environment Setup

To replicate the environment in order to run the code:

1. **Clone the repository**
   ```
   git clone https://github.com/ptsanchez/multi-asset-market-modeling-challenge.git
   cd multi-asset-market-modeling-challenge
   ```
2. **Create Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On windows, use `venv\Scripts\Activate`
   ```
3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. Create a project_data/ folder in the root of this project. Download the data archive (project_data.tar.gz) from [this link.](https://drive.google.com/drive/folders/1vunle__icJbzdtzjRJsF5j8CrPOsEyNI), and extract its contents into this folder. Your directory structure should look like this:
  ```
   /multi-asset-market-modeling-challenge
    /notebooks
      exploratory_data_analysis.ipynb
    /project_data/
        /crypto/
        /equity/
        /etf/
        /futures/
        /fx/
        /index/
    .gitignore
    requirements.txt
    README.md
    /src/
       (*.py files)
  ```

### How to Run Code

You can run different modeling pipelines by specifying a mode and optional command-lind arguments. The mode is **required** and tells the script which operation to run
- `baseline-vol`: Runs the baseline volatility prediction model. This model uses only the target asset's own lagged returns as features.
- `regime-vol`: Runs the regime-aware volatility prediction model. This model first identifies market regimes using cross-asset features and then uses the regime as a feature for volatility prediction.
- `regime-analysis`: Rins regime characteristic analysis without running full walk-forward evaluation.

  #### Command line Arguments

  You can modify the behavior of the models using the following arguments:
