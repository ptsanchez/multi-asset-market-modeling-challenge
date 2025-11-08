import pandas as pd
from pathlib import Path
from config import CRYPTO_DIR, EQUITY_DIR, ETF_DIR, FUTURES_DIR, FX_DIR, INDEX_DIR, DATA_FREQ

def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a pandas DataFrame."""
    df = pd.read_parquet(path, engine='fastparquet')
    df = df.set_index("ts")

    return df

def load_asset_group(dir_path: Path) -> dict:
    """
    Loads parquet files in a directory and returns dictionary of DataFrames.
    Keys are asset symbols derived from filenames without file extension.
    """

    data = {}
    for p in dir_path.glob("*.parquet"):
        name = p.stem.upper()  
        data[name] = load_parquet(p)
    return data

def load_all_data() -> dict:
    """
    Loads all asset classes into nested dictionary.
    Example: data["crypto"]["BTC"]
    """

    return {
        "crypto": load_asset_group(CRYPTO_DIR),
        "equity": load_asset_group(EQUITY_DIR),
        "etf": load_asset_group(ETF_DIR),
        "futures": load_asset_group(FUTURES_DIR),
        "fx": load_asset_group(FX_DIR),
        "index": load_asset_group(INDEX_DIR),
    }