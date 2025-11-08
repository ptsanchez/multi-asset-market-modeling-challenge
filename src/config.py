from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / 'project_data'

CRYPTO_DIR = DATA_DIR / 'crypto'
EQUITY_DIR = DATA_DIR / 'equity'
ETF_DIR = DATA_DIR / 'etf'
FUTURES_DIR = DATA_DIR / 'futures'
FX_DIR = DATA_DIR / 'fx'
INDEX_DIR = DATA_DIR / 'index'

DATA_FREQ = "1min"

BASELINE_LOOKBACK = 5 # number of past returns to use as features in baseline model